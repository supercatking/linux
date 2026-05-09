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
#define VIRT_LLM_REG_ABI        0x34
#define VIRT_LLM_REG_Q_MAX      0x38
#define VIRT_LLM_REG_XFER_MAX   0x3c
#define VIRT_LLM_REG_IRQ_VEC    0x40
#define VIRT_LLM_REG_Q_CTRL     0x44
#define VIRT_LLM_REG_Q_STATUS   0x48
#define VIRT_LLM_REG_Q_ERROR    0x4c
#define VIRT_LLM_REG_CQ_SIZE    0x50
#define VIRT_LLM_REG_CQ_LO      0x54
#define VIRT_LLM_REG_CQ_HI      0x58
#define VIRT_LLM_REG_CQ_HEAD    0x5c
#define VIRT_LLM_REG_CQ_TAIL    0x60

#define VIRT_LLM_MAGIC          0x4c4c4d31u /* "LLM1" */
#define VIRT_LLM_STATUS_XOR     0xa5a5a5a5u
#define VIRT_LLM_TEST_DOORBELL  0x13579bdfu

#define VIRT_LLM_FEATURE_QUEUE  BIT(0)
#define VIRT_LLM_FEATURE_MSI    BIT(1)
#define VIRT_LLM_FEATURE_MSIX   BIT(2)
#define VIRT_LLM_FEATURE_QCTRL  BIT(3)
#define VIRT_LLM_FEATURE_CQ     BIT(4)
#define VIRT_LLM_IRQ_COMPLETE   BIT(0)
#define VIRT_LLM_IRQ_ERROR      BIT(1)
#define VIRT_LLM_IRQ_ALL        (VIRT_LLM_IRQ_COMPLETE | VIRT_LLM_IRQ_ERROR)
#define VIRT_LLM_CMD_KICK       1
#define VIRT_LLM_OP_INFER       0x0001
#define VIRT_LLM_OP_DMA_COPY    0x0010
#define VIRT_LLM_OP_VEC_ADD_U32 0x0100
#define VIRT_LLM_OP_SOFTMAX_Q16 0x0101
#define VIRT_LLM_OP_POOL_MAX_U32 0x0102
#define VIRT_LLM_OP_GEMM_U32    0x0200
#define VIRT_LLM_OP_BAD_TEST    0xffff
#define VIRT_LLM_DESC_F_READY   BIT(0)
#define VIRT_LLM_DESC_COMPLETE  1
#define VIRT_LLM_DESC_UNSUPP    0x80000002u
#define VIRT_LLM_Q_CTRL_ENABLE  BIT(0)
#define VIRT_LLM_Q_CTRL_RESET   BIT(1)
#define VIRT_LLM_Q_STATUS_EN    BIT(0)
#define VIRT_LLM_Q_STATUS_ERR   BIT(1)
#define VIRT_LLM_Q_ERR_OPCODE   3
#define VIRT_LLM_QUEUE_LEN      4
#define VIRT_LLM_CQ_LEN         8
#define VIRT_LLM_TEST_LEN       64

#define VIRT_LLM_BACKEND_COMPAT 0
#define VIRT_LLM_BACKEND_DMA    1
#define VIRT_LLM_BACKEND_VECTOR 2
#define VIRT_LLM_BACKEND_TENSOR 3

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

struct virt_llm_cpl {
	__le32 command_id;
	__le32 opcode;
	__le32 backend;
	__le32 status;
	__le32 result;
	__le32 q_head;
	__le64 rsvd0;
} __packed;

struct virt_llm_dev {
	struct pci_dev *pdev;
	void __iomem *bar;
	struct completion done;
	struct virt_llm_desc *queue;
	dma_addr_t queue_dma;
	struct virt_llm_cpl *cq;
	dma_addr_t cq_dma;
	u8 *input;
	dma_addr_t input_dma;
	u8 *input_b;
	dma_addr_t input_b_dma;
	u8 *output;
	dma_addr_t output_dma;
	int irq_vectors;
	const char *irq_mode;
};

static irqreturn_t virt_llm_irq(int irq, void *data)
{
	struct virt_llm_dev *vdev = data;
	u32 status = ioread32(vdev->bar + VIRT_LLM_REG_IRQ_STS);

	if (!(status & VIRT_LLM_IRQ_ALL))
		return IRQ_NONE;

	iowrite32(status & VIRT_LLM_IRQ_ALL, vdev->bar + VIRT_LLM_REG_IRQ_STS);
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

	vdev->cq = dmam_alloc_coherent(dev,
				       sizeof(*vdev->cq) * VIRT_LLM_CQ_LEN,
				       &vdev->cq_dma, GFP_KERNEL);
	if (!vdev->cq)
		return -ENOMEM;

	vdev->input = dmam_alloc_coherent(dev, VIRT_LLM_TEST_LEN,
					  &vdev->input_dma, GFP_KERNEL);
	if (!vdev->input)
		return -ENOMEM;

	vdev->output = dmam_alloc_coherent(dev, VIRT_LLM_TEST_LEN,
					   &vdev->output_dma, GFP_KERNEL);
	if (!vdev->output)
		return -ENOMEM;

	vdev->input_b = dmam_alloc_coherent(dev, VIRT_LLM_TEST_LEN,
					    &vdev->input_b_dma, GFP_KERNEL);
	if (!vdev->input_b)
		return -ENOMEM;

	return 0;
}

static int virt_llm_check_cq(struct virt_llm_dev *vdev, u32 command_id,
			     u32 opcode, u32 backend, u32 status, u32 result)
{
	u32 cq_tail = ioread32(vdev->bar + VIRT_LLM_REG_CQ_TAIL);
	struct virt_llm_cpl *cpl;

	if (!cq_tail)
		return -EIO;

	cpl = &vdev->cq[(cq_tail - 1) % VIRT_LLM_CQ_LEN];
	if (le32_to_cpu(cpl->command_id) != command_id ||
	    le32_to_cpu(cpl->opcode) != opcode ||
	    le32_to_cpu(cpl->backend) != backend ||
	    le32_to_cpu(cpl->status) != status ||
	    le32_to_cpu(cpl->result) != result) {
		dev_err(&vdev->pdev->dev,
			"bad cq entry tail=%u cmd=%u/0x%x opcode=0x%x/0x%x backend=%u/%u status=0x%x/0x%x result=0x%x/0x%x\n",
			cq_tail, le32_to_cpu(cpl->command_id), command_id,
			le32_to_cpu(cpl->opcode), opcode,
			le32_to_cpu(cpl->backend), backend,
			le32_to_cpu(cpl->status), status,
			le32_to_cpu(cpl->result), result);
		return -EIO;
	}

	iowrite32(cq_tail, vdev->bar + VIRT_LLM_REG_CQ_HEAD);
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

static int virt_llm_submit_tail(struct virt_llm_dev *vdev, u32 tail,
				const char *label)
{
	unsigned long timeout;

	reinit_completion(&vdev->done);
	dma_wmb();
	iowrite32(tail, vdev->bar + VIRT_LLM_REG_Q_TAIL);
	iowrite32(VIRT_LLM_CMD_KICK, vdev->bar + VIRT_LLM_REG_COMMAND);

	timeout = wait_for_completion_timeout(&vdev->done, msecs_to_jiffies(5000));
	if (!timeout) {
		dev_err(&vdev->pdev->dev,
			"%s timed out irq_status=0x%08x head=%u tail=%u\n",
			label, ioread32(vdev->bar + VIRT_LLM_REG_IRQ_STS),
			ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD),
			ioread32(vdev->bar + VIRT_LLM_REG_Q_TAIL));
		return -ETIMEDOUT;
	}

	dma_rmb();
	return 0;
}

static int virt_llm_run_dma_selftest(struct virt_llm_dev *vdev)
{
	struct virt_llm_desc *desc = &vdev->queue[0];
	u32 features = ioread32(vdev->bar + VIRT_LLM_REG_FEATURES);
	u32 expected_sum = 0;
	int ret;

	if (!(features & VIRT_LLM_FEATURE_QUEUE))
		return -EOPNOTSUPP;
	if (!(features & VIRT_LLM_FEATURE_QCTRL))
		return -EOPNOTSUPP;
	if (!(features & VIRT_LLM_FEATURE_CQ))
		return -EOPNOTSUPP;

	for (int i = 0; i < VIRT_LLM_TEST_LEN; i++) {
		vdev->input[i] = i;
		vdev->output[i] = 0;
		expected_sum += (u8)(vdev->input[i] ^ 0x5a);
	}

	memset(vdev->queue, 0, sizeof(*vdev->queue) * VIRT_LLM_QUEUE_LEN);
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_INFER);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->len = cpu_to_le32(VIRT_LLM_TEST_LEN);
	desc->rsvd0 = cpu_to_le32(1);

	iowrite32(lower_32_bits(vdev->queue_dma), vdev->bar + VIRT_LLM_REG_Q_LO);
	iowrite32(upper_32_bits(vdev->queue_dma), vdev->bar + VIRT_LLM_REG_Q_HI);
	iowrite32(VIRT_LLM_QUEUE_LEN, vdev->bar + VIRT_LLM_REG_Q_SIZE);
	iowrite32(lower_32_bits(vdev->cq_dma), vdev->bar + VIRT_LLM_REG_CQ_LO);
	iowrite32(upper_32_bits(vdev->cq_dma), vdev->bar + VIRT_LLM_REG_CQ_HI);
	iowrite32(VIRT_LLM_CQ_LEN, vdev->bar + VIRT_LLM_REG_CQ_SIZE);
	iowrite32(VIRT_LLM_IRQ_ALL, vdev->bar + VIRT_LLM_REG_IRQ_MASK);
	iowrite32(VIRT_LLM_Q_CTRL_ENABLE, vdev->bar + VIRT_LLM_REG_Q_CTRL);
	if (!(ioread32(vdev->bar + VIRT_LLM_REG_Q_STATUS) & VIRT_LLM_Q_STATUS_EN))
		return -EIO;

	ret = virt_llm_submit_tail(vdev, 1, "xor inference self-test");
	if (ret)
		return ret;
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
	ret = virt_llm_check_cq(vdev, 1, VIRT_LLM_OP_INFER,
				VIRT_LLM_BACKEND_COMPAT, VIRT_LLM_DESC_COMPLETE,
				expected_sum);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev,
		 "dma inference ok: irq=%s queue=%pad input=%pad output=%pad len=%u checksum=0x%08x\n",
		 vdev->irq_mode,
		 &vdev->queue_dma, &vdev->input_dma, &vdev->output_dma,
		 VIRT_LLM_TEST_LEN, expected_sum);

	return 0;
}

static int virt_llm_run_dma_copy_selftest(struct virt_llm_dev *vdev)
{
	struct virt_llm_desc *desc = &vdev->queue[1];
	u32 expected_sum = 0;
	int ret;

	for (int i = 0; i < VIRT_LLM_TEST_LEN; i++) {
		vdev->input[i] = 0xa0 + i;
		vdev->output[i] = 0;
		expected_sum += vdev->input[i];
	}

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_DMA_COPY);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->len = cpu_to_le32(VIRT_LLM_TEST_LEN);
	desc->rsvd0 = cpu_to_le32(2);

	ret = virt_llm_submit_tail(vdev, 2, "dma copy self-test");
	if (ret)
		return ret;

	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE ||
	    le32_to_cpu(desc->result) != expected_sum) {
		dev_err(&vdev->pdev->dev,
			"dma copy bad status=0x%08x result=0x%08x expected=0x%08x\n",
			le32_to_cpu(desc->status), le32_to_cpu(desc->result),
			expected_sum);
		return -EIO;
	}

	for (int i = 0; i < VIRT_LLM_TEST_LEN; i++) {
		if (vdev->output[i] != vdev->input[i]) {
			dev_err(&vdev->pdev->dev,
				"dma copy mismatch at %d got=0x%02x expected=0x%02x\n",
				i, vdev->output[i], vdev->input[i]);
			return -EIO;
		}
	}
	ret = virt_llm_check_cq(vdev, 2, VIRT_LLM_OP_DMA_COPY,
				VIRT_LLM_BACKEND_DMA, VIRT_LLM_DESC_COMPLETE,
				expected_sum);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev, "dma copy ok: len=%u checksum=0x%08x\n",
		 VIRT_LLM_TEST_LEN, expected_sum);
	return 0;
}

static int virt_llm_run_vector_add_selftest(struct virt_llm_dev *vdev)
{
	struct virt_llm_desc *desc = &vdev->queue[2];
	__le32 *a = (__le32 *)vdev->input;
	__le32 *b = (__le32 *)vdev->input_b;
	__le32 *out = (__le32 *)vdev->output;
	u32 count = 8;
	u32 expected_sum = 0;
	int ret;

	memset(vdev->output, 0, VIRT_LLM_TEST_LEN);
	for (int i = 0; i < count; i++) {
		u32 av = i + 1;
		u32 bv = 100 + i;
		u32 sum = av + bv;

		a[i] = cpu_to_le32(av);
		b[i] = cpu_to_le32(bv);
		expected_sum += sum;
	}

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_VEC_ADD_U32);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->len = cpu_to_le32(count);
	desc->rsvd0 = cpu_to_le32(3);
	desc->rsvd1 = cpu_to_le64(vdev->input_b_dma);

	ret = virt_llm_submit_tail(vdev, 3, "vector add self-test");
	if (ret)
		return ret;

	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE ||
	    le32_to_cpu(desc->result) != expected_sum) {
		dev_err(&vdev->pdev->dev,
			"vector add bad status=0x%08x result=0x%08x expected=0x%08x\n",
			le32_to_cpu(desc->status), le32_to_cpu(desc->result),
			expected_sum);
		return -EIO;
	}

	for (int i = 0; i < count; i++) {
		u32 expected = (i + 1) + (100 + i);

		if (le32_to_cpu(out[i]) != expected) {
			dev_err(&vdev->pdev->dev,
				"vector add mismatch at %d got=%u expected=%u\n",
				i, le32_to_cpu(out[i]), expected);
			return -EIO;
		}
	}
	ret = virt_llm_check_cq(vdev, 3, VIRT_LLM_OP_VEC_ADD_U32,
				VIRT_LLM_BACKEND_VECTOR,
				VIRT_LLM_DESC_COMPLETE, expected_sum);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev, "vector add ok: count=%u checksum=0x%08x\n",
		 count, expected_sum);
	return 0;
}

static int virt_llm_run_gemm_selftest(struct virt_llm_dev *vdev)
{
	u32 head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	u32 idx = head % VIRT_LLM_QUEUE_LEN;
	struct virt_llm_desc *desc = &vdev->queue[idx];
	__le32 *a = (__le32 *)vdev->input;
	__le32 *b = (__le32 *)vdev->input_b;
	__le32 *out = (__le32 *)vdev->output;
	u32 expected[] = { 19, 22, 43, 50 };
	u64 dims = 2 | (2ULL << 16) | (2ULL << 32);
	u32 expected_sum = 0;
	int ret;

	a[0] = cpu_to_le32(1);
	a[1] = cpu_to_le32(2);
	a[2] = cpu_to_le32(3);
	a[3] = cpu_to_le32(4);
	b[0] = cpu_to_le32(5);
	b[1] = cpu_to_le32(6);
	b[2] = cpu_to_le32(7);
	b[3] = cpu_to_le32(8);
	memset(vdev->output, 0, VIRT_LLM_TEST_LEN);
	for (int i = 0; i < ARRAY_SIZE(expected); i++)
		expected_sum += expected[i];

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_GEMM_U32);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->rsvd0 = cpu_to_le32(6);
	desc->rsvd1 = cpu_to_le64(vdev->input_b_dma);
	desc->rsvd2 = cpu_to_le64(dims);

	ret = virt_llm_submit_tail(vdev, head + 1, "gemm self-test");
	if (ret)
		return ret;

	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE ||
	    le32_to_cpu(desc->result) != expected_sum) {
		dev_err(&vdev->pdev->dev,
			"gemm bad status=0x%08x result=0x%08x expected=0x%08x\n",
			le32_to_cpu(desc->status), le32_to_cpu(desc->result),
			expected_sum);
		return -EIO;
	}

	for (int i = 0; i < ARRAY_SIZE(expected); i++) {
		if (le32_to_cpu(out[i]) != expected[i]) {
			dev_err(&vdev->pdev->dev,
				"gemm mismatch at %d got=%u expected=%u\n",
				i, le32_to_cpu(out[i]), expected[i]);
			return -EIO;
		}
	}
	ret = virt_llm_check_cq(vdev, 6, VIRT_LLM_OP_GEMM_U32,
				VIRT_LLM_BACKEND_TENSOR,
				VIRT_LLM_DESC_COMPLETE, expected_sum);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev, "gemm ok: m=2 n=2 k=2 checksum=0x%08x\n",
		 expected_sum);
	return 0;
}

static int virt_llm_run_softmax_selftest(struct virt_llm_dev *vdev)
{
	u32 head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	u32 idx = head % VIRT_LLM_QUEUE_LEN;
	struct virt_llm_desc *desc = &vdev->queue[idx];
	__le32 *input = (__le32 *)vdev->input;
	__le32 *out = (__le32 *)vdev->output;
	u32 expected[] = { 8192, 8192, 16384, 32768 };
	u32 values[] = { 1, 1, 2, 4 };
	u32 expected_sum = 0;
	int ret;

	memset(vdev->output, 0, VIRT_LLM_TEST_LEN);
	for (int i = 0; i < ARRAY_SIZE(values); i++) {
		input[i] = cpu_to_le32(values[i]);
		expected_sum += expected[i];
	}

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_SOFTMAX_Q16);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->len = cpu_to_le32(ARRAY_SIZE(values));
	desc->rsvd0 = cpu_to_le32(4);

	ret = virt_llm_submit_tail(vdev, head + 1, "softmax self-test");
	if (ret)
		return ret;

	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE ||
	    le32_to_cpu(desc->result) != expected_sum) {
		dev_err(&vdev->pdev->dev,
			"softmax bad status=0x%08x result=0x%08x expected=0x%08x\n",
			le32_to_cpu(desc->status), le32_to_cpu(desc->result),
			expected_sum);
		return -EIO;
	}

	for (int i = 0; i < ARRAY_SIZE(expected); i++) {
		if (le32_to_cpu(out[i]) != expected[i]) {
			dev_err(&vdev->pdev->dev,
				"softmax mismatch at %d got=%u expected=%u\n",
				i, le32_to_cpu(out[i]), expected[i]);
			return -EIO;
		}
	}

	ret = virt_llm_check_cq(vdev, 4, VIRT_LLM_OP_SOFTMAX_Q16,
				VIRT_LLM_BACKEND_VECTOR,
				VIRT_LLM_DESC_COMPLETE, expected_sum);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev, "softmax q16 ok: count=%zu checksum=0x%08x\n",
		 ARRAY_SIZE(values), expected_sum);
	return 0;
}

static int virt_llm_run_pooling_selftest(struct virt_llm_dev *vdev)
{
	u32 head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	u32 idx = head % VIRT_LLM_QUEUE_LEN;
	struct virt_llm_desc *desc = &vdev->queue[idx];
	__le32 *input = (__le32 *)vdev->input;
	__le32 *out = (__le32 *)vdev->output;
	u32 values[] = { 3, 1, 7, 6, 2, 9, 4, 5 };
	u32 expected[] = { 3, 7, 9, 5 };
	u32 expected_sum = 0;
	u64 args = 2;
	int ret;

	memset(vdev->output, 0, VIRT_LLM_TEST_LEN);
	for (int i = 0; i < ARRAY_SIZE(values); i++)
		input[i] = cpu_to_le32(values[i]);
	for (int i = 0; i < ARRAY_SIZE(expected); i++)
		expected_sum += expected[i];

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_POOL_MAX_U32);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->len = cpu_to_le32(ARRAY_SIZE(values));
	desc->rsvd0 = cpu_to_le32(5);
	desc->rsvd2 = cpu_to_le64(args);

	ret = virt_llm_submit_tail(vdev, head + 1, "pooling self-test");
	if (ret)
		return ret;

	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE ||
	    le32_to_cpu(desc->result) != expected_sum) {
		dev_err(&vdev->pdev->dev,
			"pooling bad status=0x%08x result=0x%08x expected=0x%08x\n",
			le32_to_cpu(desc->status), le32_to_cpu(desc->result),
			expected_sum);
		return -EIO;
	}

	for (int i = 0; i < ARRAY_SIZE(expected); i++) {
		if (le32_to_cpu(out[i]) != expected[i]) {
			dev_err(&vdev->pdev->dev,
				"pooling mismatch at %d got=%u expected=%u\n",
				i, le32_to_cpu(out[i]), expected[i]);
			return -EIO;
		}
	}

	ret = virt_llm_check_cq(vdev, 5, VIRT_LLM_OP_POOL_MAX_U32,
				VIRT_LLM_BACKEND_VECTOR,
				VIRT_LLM_DESC_COMPLETE, expected_sum);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev, "pool max ok: count=%zu window=2 checksum=0x%08x\n",
		 ARRAY_SIZE(values), expected_sum);
	return 0;
}

static int virt_llm_run_error_selftest(struct virt_llm_dev *vdev)
{
	u32 head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	u32 idx = head % VIRT_LLM_QUEUE_LEN;
	struct virt_llm_desc *desc = &vdev->queue[idx];
	u32 q_error;
	u32 q_status;
	int ret;

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_BAD_TEST);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->rsvd0 = cpu_to_le32(7);

	ret = virt_llm_submit_tail(vdev, head + 1, "error self-test");
	if (ret)
		return ret;

	q_status = ioread32(vdev->bar + VIRT_LLM_REG_Q_STATUS);
	q_error = ioread32(vdev->bar + VIRT_LLM_REG_Q_ERROR);
	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_UNSUPP ||
	    !(q_status & VIRT_LLM_Q_STATUS_ERR) ||
	    q_error != VIRT_LLM_Q_ERR_OPCODE) {
		dev_err(&vdev->pdev->dev,
			"bad error self-test desc=0x%08x q_status=0x%08x q_error=%u\n",
			le32_to_cpu(desc->status), q_status, q_error);
		return -EIO;
	}
	ret = virt_llm_check_cq(vdev, 7, VIRT_LLM_OP_BAD_TEST, U32_MAX,
				VIRT_LLM_DESC_UNSUPP, 0);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev,
		 "error path ok: desc_status=0x%08x q_status=0x%08x q_error=%u\n",
		 le32_to_cpu(desc->status), q_status, q_error);
	iowrite32(VIRT_LLM_Q_CTRL_RESET, vdev->bar + VIRT_LLM_REG_Q_CTRL);
	return 0;
}

static int virt_llm_pci_probe(struct pci_dev *pdev,
			      const struct pci_device_id *id)
{
	struct virt_llm_dev *vdev;
	void __iomem *bar;
	u32 magic;
	u32 version;
	u32 abi;
	u32 q_max;
	u32 xfer_max;
	u32 irq_vec;
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
	abi = ioread32(bar + VIRT_LLM_REG_ABI);
	q_max = ioread32(bar + VIRT_LLM_REG_Q_MAX);
	xfer_max = ioread32(bar + VIRT_LLM_REG_XFER_MAX);
	irq_vec = ioread32(bar + VIRT_LLM_REG_IRQ_VEC);
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
	dev_info(&pdev->dev, "dma buffers: queue=%pad cq=%pad input=%pad output=%pad\n",
		 &vdev->queue_dma, &vdev->cq_dma, &vdev->input_dma,
		 &vdev->output_dma);
	dev_info(&pdev->dev, "dma input_b=%pad\n", &vdev->input_b_dma);

	ret = virt_llm_request_irq(vdev);
	if (ret)
		return ret;

	ret = virt_llm_run_dma_selftest(vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		return ret;
	}

	ret = virt_llm_run_dma_copy_selftest(vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		return ret;
	}

	ret = virt_llm_run_vector_add_selftest(vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		return ret;
	}

	ret = virt_llm_run_softmax_selftest(vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		return ret;
	}

	ret = virt_llm_run_pooling_selftest(vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		return ret;
	}

	ret = virt_llm_run_gemm_selftest(vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		return ret;
	}

	ret = virt_llm_run_error_selftest(vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		return ret;
	}

	dev_info(&pdev->dev,
		 "probe ok: magic=0x%08x version=%u abi=%u q_max=%u xfer_max=%u irq_vec=%u doorbell=0x%08x status=0x%08x\n",
		 magic, version, abi, q_max, xfer_max, irq_vec,
		 VIRT_LLM_TEST_DOORBELL, status);

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
