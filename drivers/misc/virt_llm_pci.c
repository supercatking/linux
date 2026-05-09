// SPDX-License-Identifier: GPL-2.0-only
/*
 * Minimal PCI driver for the QEMU virt-llm prototype device.
 */

#include <linux/io.h>
#include <linux/completion.h>
#include <linux/dma-mapping.h>
#include <linux/interrupt.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/slab.h>

#define VIRT_LLM_DEVICE_ID      0x1100

#define VIRT_LLM_REG_MAGIC      0x00
#define VIRT_LLM_REG_VERSION    0x04
#define VIRT_LLM_REG_DOORBELL   0x08
#define VIRT_LLM_REG_STATUS     0x0c
#define VIRT_LLM_REG_FEATURES   0x10
#define VIRT_LLM_REG_Q_SIZE     0x14
#define VIRT_LLM_REG_Q_LO       0x18
#define VIRT_LLM_REG_Q_HI       0x1c
#define VIRT_LLM_REG_Q_HEAD     0x20
#define VIRT_LLM_REG_Q_TAIL     0x24
#define VIRT_LLM_REG_IRQ_STS    0x28
#define VIRT_LLM_REG_IRQ_MASK   0x2c
#define VIRT_LLM_REG_COMMAND    0x30

#define VIRT_LLM_MAGIC          0x4c4c4d31u /* "LLM1" */
#define VIRT_LLM_STATUS_XOR     0xa5a5a5a5u
#define VIRT_LLM_TEST_DOORBELL  0x13579bdfu

#define VIRT_LLM_FEATURE_QUEUE  BIT(0)
#define VIRT_LLM_FEATURE_MSI    BIT(1)
#define VIRT_LLM_FEATURE_MSIX   BIT(2)
#define VIRT_LLM_IRQ_COMPLETE   BIT(0)
#define VIRT_LLM_CMD_KICK       1
#define VIRT_LLM_OP_INFER       1
#define VIRT_LLM_DESC_COMPLETE  1
#define VIRT_LLM_QUEUE_LEN      4
#define VIRT_LLM_TEST_LEN       64

struct virt_llm_desc {
	__le32 opcode;
	__le32 flags;
	__le64 input_addr;
	__le64 output_addr;
	__le32 len;
	__le32 status;
	__le32 result;
	__le32 rsvd0;
	__le64 rsvd1;
	__le64 rsvd2;
	__le64 rsvd3;
} __packed;

struct virt_llm_dev {
	struct pci_dev *pdev;
	void __iomem *bar;
	struct completion done;
	struct virt_llm_desc *queue;
	dma_addr_t queue_dma;
	u8 *input;
	dma_addr_t input_dma;
	u8 *output;
	dma_addr_t output_dma;
	int irq_vectors;
	const char *irq_mode;
};

static irqreturn_t virt_llm_irq(int irq, void *data)
{
	struct virt_llm_dev *vdev = data;
	u32 status = ioread32(vdev->bar + VIRT_LLM_REG_IRQ_STS);

	if (!(status & VIRT_LLM_IRQ_COMPLETE))
		return IRQ_NONE;

	iowrite32(VIRT_LLM_IRQ_COMPLETE, vdev->bar + VIRT_LLM_REG_IRQ_STS);
	complete(&vdev->done);
	return IRQ_HANDLED;
}

static int virt_llm_alloc_dma(struct virt_llm_dev *vdev)
{
	struct device *dev = &vdev->pdev->dev;

	vdev->queue = dmam_alloc_coherent(dev,
					  sizeof(*vdev->queue) * VIRT_LLM_QUEUE_LEN,
					  &vdev->queue_dma, GFP_KERNEL);
	if (!vdev->queue)
		return -ENOMEM;

	vdev->input = dmam_alloc_coherent(dev, VIRT_LLM_TEST_LEN,
					  &vdev->input_dma, GFP_KERNEL);
	if (!vdev->input)
		return -ENOMEM;

	vdev->output = dmam_alloc_coherent(dev, VIRT_LLM_TEST_LEN,
					   &vdev->output_dma, GFP_KERNEL);
	if (!vdev->output)
		return -ENOMEM;

	return 0;
}

static int virt_llm_request_irq(struct virt_llm_dev *vdev)
{
	struct pci_dev *pdev = vdev->pdev;
	int ret;

	ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSIX | PCI_IRQ_MSI);
	if (ret < 0) {
		ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_INTX);
		if (ret < 0)
			return ret;
		vdev->irq_mode = "intx";
	} else {
		vdev->irq_mode = pdev->msix_enabled ? "msix" : "msi";
	}

	vdev->irq_vectors = ret;
	ret = devm_request_irq(&pdev->dev, pci_irq_vector(pdev, 0),
			       virt_llm_irq, 0, "virt_llm_pci", vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		vdev->irq_vectors = 0;
	}

	return ret;
}

static int virt_llm_run_dma_selftest(struct virt_llm_dev *vdev)
{
	struct virt_llm_desc *desc = &vdev->queue[0];
	u32 features = ioread32(vdev->bar + VIRT_LLM_REG_FEATURES);
	u32 expected_sum = 0;
	unsigned long timeout;

	if (!(features & VIRT_LLM_FEATURE_QUEUE))
		return -EOPNOTSUPP;

	for (int i = 0; i < VIRT_LLM_TEST_LEN; i++) {
		vdev->input[i] = i;
		vdev->output[i] = 0;
		expected_sum += (u8)(vdev->input[i] ^ 0x5a);
	}

	memset(vdev->queue, 0, sizeof(*vdev->queue) * VIRT_LLM_QUEUE_LEN);
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_INFER);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->len = cpu_to_le32(VIRT_LLM_TEST_LEN);

	reinit_completion(&vdev->done);
	iowrite32(lower_32_bits(vdev->queue_dma), vdev->bar + VIRT_LLM_REG_Q_LO);
	iowrite32(upper_32_bits(vdev->queue_dma), vdev->bar + VIRT_LLM_REG_Q_HI);
	iowrite32(VIRT_LLM_QUEUE_LEN, vdev->bar + VIRT_LLM_REG_Q_SIZE);
	iowrite32(VIRT_LLM_IRQ_COMPLETE, vdev->bar + VIRT_LLM_REG_IRQ_MASK);

	dma_wmb();
	iowrite32(1, vdev->bar + VIRT_LLM_REG_Q_TAIL);
	iowrite32(VIRT_LLM_CMD_KICK, vdev->bar + VIRT_LLM_REG_COMMAND);

	timeout = wait_for_completion_timeout(&vdev->done, msecs_to_jiffies(5000));
	if (!timeout) {
		dev_err(&vdev->pdev->dev, "dma self-test timed out irq_status=0x%08x head=%u tail=%u\n",
			ioread32(vdev->bar + VIRT_LLM_REG_IRQ_STS),
			ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD),
			ioread32(vdev->bar + VIRT_LLM_REG_Q_TAIL));
		return -ETIMEDOUT;
	}

	dma_rmb();
	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE) {
		dev_err(&vdev->pdev->dev, "bad descriptor status=0x%08x result=0x%08x\n",
			le32_to_cpu(desc->status), le32_to_cpu(desc->result));
		return -EIO;
	}

	for (int i = 0; i < VIRT_LLM_TEST_LEN; i++) {
		if (vdev->output[i] != (u8)(vdev->input[i] ^ 0x5a)) {
			dev_err(&vdev->pdev->dev, "output mismatch at %d got=0x%02x expected=0x%02x\n",
				i, vdev->output[i], (u8)(vdev->input[i] ^ 0x5a));
			return -EIO;
		}
	}

	if (le32_to_cpu(desc->result) != expected_sum) {
		dev_err(&vdev->pdev->dev, "checksum mismatch got=0x%08x expected=0x%08x\n",
			le32_to_cpu(desc->result), expected_sum);
		return -EIO;
	}

	dev_info(&vdev->pdev->dev,
		 "dma inference ok: irq=%s queue=%pad input=%pad output=%pad len=%u checksum=0x%08x\n",
		 vdev->irq_mode,
		 &vdev->queue_dma, &vdev->input_dma, &vdev->output_dma,
		 VIRT_LLM_TEST_LEN, expected_sum);

	return 0;
}

static int virt_llm_pci_probe(struct pci_dev *pdev,
			      const struct pci_device_id *id)
{
	struct virt_llm_dev *vdev;
	void __iomem *bar;
	u32 magic;
	u32 version;
	u32 status;
	u32 expected;
	int ret;

	ret = pcim_enable_device(pdev);
	if (ret)
		return ret;
	pci_set_master(pdev);

	ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
	if (ret)
		return ret;

	ret = pcim_iomap_regions(pdev, BIT(0), "virt_llm_pci");
	if (ret)
		return ret;

	bar = pcim_iomap_table(pdev)[0];
	if (!bar)
		return -ENOMEM;

	magic = ioread32(bar + VIRT_LLM_REG_MAGIC);
	version = ioread32(bar + VIRT_LLM_REG_VERSION);
	if (magic != VIRT_LLM_MAGIC) {
		dev_err(&pdev->dev, "bad magic: 0x%08x\n", magic);
		return -ENODEV;
	}

	iowrite32(VIRT_LLM_TEST_DOORBELL, bar + VIRT_LLM_REG_DOORBELL);
	status = ioread32(bar + VIRT_LLM_REG_STATUS);
	expected = VIRT_LLM_TEST_DOORBELL ^ VIRT_LLM_STATUS_XOR;
	if (status != expected) {
		dev_err(&pdev->dev, "doorbell self-test failed: status=0x%08x expected=0x%08x\n",
			status, expected);
		return -EIO;
	}

	vdev = devm_kzalloc(&pdev->dev, sizeof(*vdev), GFP_KERNEL);
	if (!vdev)
		return -ENOMEM;

	vdev->pdev = pdev;
	vdev->bar = bar;
	init_completion(&vdev->done);
	pci_set_drvdata(pdev, vdev);

	ret = virt_llm_alloc_dma(vdev);
	if (ret)
		return ret;
	dev_info(&pdev->dev, "dma buffers: queue=%pad input=%pad output=%pad\n",
		 &vdev->queue_dma, &vdev->input_dma, &vdev->output_dma);

	ret = virt_llm_request_irq(vdev);
	if (ret)
		return ret;

	ret = virt_llm_run_dma_selftest(vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		return ret;
	}

	dev_info(&pdev->dev,
		 "probe ok: magic=0x%08x version=%u doorbell=0x%08x status=0x%08x\n",
		 magic, version, VIRT_LLM_TEST_DOORBELL, status);

	return 0;
}

static void virt_llm_pci_remove(struct pci_dev *pdev)
{
	pci_free_irq_vectors(pdev);
	dev_info(&pdev->dev, "remove\n");
}

static const struct pci_device_id virt_llm_pci_ids[] = {
	{ PCI_DEVICE(PCI_VENDOR_ID_REDHAT, VIRT_LLM_DEVICE_ID) },
	{ }
};
MODULE_DEVICE_TABLE(pci, virt_llm_pci_ids);

static struct pci_driver virt_llm_pci_driver = {
	.name = "virt_llm_pci",
	.id_table = virt_llm_pci_ids,
	.probe = virt_llm_pci_probe,
	.remove = virt_llm_pci_remove,
};

module_pci_driver(virt_llm_pci_driver);

MODULE_AUTHOR("Codex");
MODULE_DESCRIPTION("QEMU virt-llm PCI prototype driver");
MODULE_LICENSE("GPL");
