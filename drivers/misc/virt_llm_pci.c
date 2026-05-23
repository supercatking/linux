// SPDX-License-Identifier: GPL-2.0-only
/*
 * Minimal PCI driver for the QEMU virt-llm prototype device.
 */

#include <linux/io.h>
#include <linux/completion.h>
#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/interrupt.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <uapi/linux/virt_llm.h>

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
#define VIRT_LLM_REG_SCALAR_STATUS      0x64
#define VIRT_LLM_REG_SCALAR_KERNELS     0x68
#define VIRT_LLM_REG_SCALAR_LAST_KERNEL 0x6c
#define VIRT_LLM_REG_SCALAR_LAST_OPCODE 0x70
#define VIRT_LLM_REG_KERNEL_INDEX       0x74
#define VIRT_LLM_REG_KERNEL_ID          0x78
#define VIRT_LLM_REG_KERNEL_OPCODE      0x7c
#define VIRT_LLM_REG_KERNEL_ABI         0x80
#define VIRT_LLM_REG_KERNEL_ENTRY       0x84
#define VIRT_LLM_REG_KERNEL_SIZE        0x88
#define VIRT_LLM_REG_KERNEL_CHECKSUM    0x8c

#define VIRT_LLM_MAGIC          0x4c4c4d31u /* "LLM1" */
#define VIRT_LLM_STATUS_XOR     0xa5a5a5a5u
#define VIRT_LLM_TEST_DOORBELL  0x13579bdfu

#define VIRT_LLM_FEATURE_QUEUE  BIT(0)
#define VIRT_LLM_FEATURE_MSI    BIT(1)
#define VIRT_LLM_FEATURE_MSIX   BIT(2)
#define VIRT_LLM_FEATURE_QCTRL  BIT(3)
#define VIRT_LLM_FEATURE_CQ     BIT(4)
#define VIRT_LLM_FEATURE_SCALAR BIT(5)
#define VIRT_LLM_IRQ_COMPLETE   BIT(0)
#define VIRT_LLM_IRQ_ERROR      BIT(1)
#define VIRT_LLM_IRQ_ALL        (VIRT_LLM_IRQ_COMPLETE | VIRT_LLM_IRQ_ERROR)
#define VIRT_LLM_CMD_KICK       1
#define VIRT_LLM_OP_INFER       0x0001
#define VIRT_LLM_OP_DMA_COPY    0x0010
#define VIRT_LLM_OP_VEC_ADD_U32 0x0100
#define VIRT_LLM_OP_SOFTMAX_Q16 0x0101
#define VIRT_LLM_OP_POOL_MAX_U32 0x0102
#define VIRT_LLM_OP_DOT_U32     0x0103
#define VIRT_LLM_OP_GEMM_U32    0x0200
#define VIRT_LLM_OP_CONV2D_U32  0x0201
#define VIRT_LLM_OP_ATTENTION_Q16 0x0202
#define VIRT_LLM_OP_BAD_TEST    0xffff
#ifndef VIRT_LLM_DESC_F_READY
#define VIRT_LLM_DESC_F_READY   BIT(0)
#endif
#define VIRT_LLM_DESC_COMPLETE  1
#ifndef VIRT_LLM_DESC_UNSUPP
#define VIRT_LLM_DESC_UNSUPP    0x80000002u
#endif
#ifndef VIRT_LLM_DESC_BAD_KERNEL
#define VIRT_LLM_DESC_BAD_KERNEL 0x80000004u
#endif
#define VIRT_LLM_Q_CTRL_ENABLE  BIT(0)
#define VIRT_LLM_Q_CTRL_RESET   BIT(1)
#define VIRT_LLM_Q_STATUS_EN    BIT(0)
#define VIRT_LLM_Q_STATUS_ERR   BIT(1)
#define VIRT_LLM_Q_ERR_OPCODE   3
#define VIRT_LLM_Q_ERR_KERNEL   4
#define VIRT_LLM_QUEUE_LEN      4
#define VIRT_LLM_CQ_LEN         8
#define VIRT_LLM_TEST_LEN       64
#define VIRT_LLM_USER_BUF_MAX   8
#define VIRT_LLM_USER_BUF_SIZE  PAGE_SIZE
#define VIRT_LLM_MISC_MINOR     243
#ifndef VIRT_LLM_ATTENTION_CAUSAL
#define VIRT_LLM_ATTENTION_CAUSAL BIT(0)
#endif

#define VIRT_LLM_BACKEND_COMPAT 0
#define VIRT_LLM_BACKEND_DMA    1
#define VIRT_LLM_BACKEND_VECTOR 2
#define VIRT_LLM_BACKEND_TENSOR 3
#define VIRT_LLM_BACKEND_SCALAR 4

#define VIRT_LLM_KERNEL_VEC_ADD_U32 1
#define VIRT_LLM_KERNEL_DOT_U32 2
#define VIRT_LLM_KERNEL_SOFTMAX_Q16 3
#define VIRT_LLM_KERNEL_POOL_MAX_U32 4
#define VIRT_LLM_KERNEL_ABI_VERSION 1

struct virt_llm_kernel_expected {
	u32 kernel_id;
	u32 opcode;
	u32 abi;
	u32 entry;
	u32 size;
	u32 checksum;
};

static const struct virt_llm_kernel_expected virt_llm_expected_kernels[] = {
	{
		.kernel_id = VIRT_LLM_KERNEL_VEC_ADD_U32,
		.opcode = VIRT_LLM_OP_VEC_ADD_U32,
		.abi = VIRT_LLM_KERNEL_ABI_VERSION,
		.entry = 0x1000,
		.size = 64,
		.checksum = 0xadd00101,
	},
	{
		.kernel_id = VIRT_LLM_KERNEL_DOT_U32,
		.opcode = VIRT_LLM_OP_DOT_U32,
		.abi = VIRT_LLM_KERNEL_ABI_VERSION,
		.entry = 0x1100,
		.size = 72,
		.checksum = 0xd0700103,
	},
	{
		.kernel_id = VIRT_LLM_KERNEL_SOFTMAX_Q16,
		.opcode = VIRT_LLM_OP_SOFTMAX_Q16,
		.abi = VIRT_LLM_KERNEL_ABI_VERSION,
		.entry = 0x1200,
		.size = 96,
		.checksum = 0x50170101,
	},
	{
		.kernel_id = VIRT_LLM_KERNEL_POOL_MAX_U32,
		.opcode = VIRT_LLM_OP_POOL_MAX_U32,
		.abi = VIRT_LLM_KERNEL_ABI_VERSION,
		.entry = 0x1300,
		.size = 80,
		.checksum = 0x90010102,
	},
};

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

struct virt_llm_user_dma {
	void *cpu;
	dma_addr_t dma;
	size_t size;
	bool in_use;
};

struct virt_llm_dev {
	struct pci_dev *pdev;
	void __iomem *bar;
	struct completion done;
	struct mutex user_lock;
	struct virt_llm_desc *queue;
	dma_addr_t queue_dma;
	struct virt_llm_cpl *cq;
	dma_addr_t cq_dma;
	u8 *input;
	dma_addr_t input_dma;
	u8 *input_b;
	dma_addr_t input_b_dma;
	u8 *input_c;
	dma_addr_t input_c_dma;
	u8 *output;
	dma_addr_t output_dma;
	int irq_vectors;
	const char *irq_mode;
	struct miscdevice miscdev;
	u32 user_cq_head;
	u32 next_command_id;
	struct virt_llm_user_dma user_bufs[VIRT_LLM_USER_BUF_MAX];
};

static bool run_selftest = true;
module_param(run_selftest, bool, 0644);
MODULE_PARM_DESC(run_selftest, "Run virt-llm probe selftests");

static struct virt_llm_dev *virt_llm_singleton;

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

	vdev->input_c = dmam_alloc_coherent(dev, VIRT_LLM_TEST_LEN,
					    &vdev->input_c_dma, GFP_KERNEL);
	if (!vdev->input_c)
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
	if (!(features & VIRT_LLM_FEATURE_SCALAR))
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
	desc->rsvd3 = cpu_to_le64(VIRT_LLM_KERNEL_VEC_ADD_U32);

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
				VIRT_LLM_BACKEND_SCALAR,
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

static int virt_llm_run_dot_selftest(struct virt_llm_dev *vdev)
{
	u32 head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	u32 idx = head % VIRT_LLM_QUEUE_LEN;
	struct virt_llm_desc *desc = &vdev->queue[idx];
	__le32 *a = (__le32 *)vdev->input;
	__le32 *b = (__le32 *)vdev->input_b;
	u32 avec[] = { 1, 2, 3, 4 };
	u32 bvec[] = { 5, 6, 7, 8 };
	u32 expected = 70;
	int ret;

	for (int i = 0; i < ARRAY_SIZE(avec); i++) {
		a[i] = cpu_to_le32(avec[i]);
		b[i] = cpu_to_le32(bvec[i]);
	}

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_DOT_U32);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->len = cpu_to_le32(ARRAY_SIZE(avec));
	desc->rsvd0 = cpu_to_le32(8);
	desc->rsvd1 = cpu_to_le64(vdev->input_b_dma);
	desc->rsvd3 = cpu_to_le64(VIRT_LLM_KERNEL_DOT_U32);

	ret = virt_llm_submit_tail(vdev, head + 1, "dot self-test");
	if (ret)
		return ret;

	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE ||
	    le32_to_cpu(desc->result) != expected) {
		dev_err(&vdev->pdev->dev,
			"dot bad status=0x%08x result=%u expected=%u\n",
			le32_to_cpu(desc->status), le32_to_cpu(desc->result),
			expected);
		return -EIO;
	}

	ret = virt_llm_check_cq(vdev, 8, VIRT_LLM_OP_DOT_U32,
				VIRT_LLM_BACKEND_SCALAR,
				VIRT_LLM_DESC_COMPLETE, expected);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev, "dot u32 ok: count=%zu result=%u\n",
		 ARRAY_SIZE(avec), expected);
	return 0;
}

static int virt_llm_run_conv2d_selftest(struct virt_llm_dev *vdev)
{
	u32 head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	u32 idx = head % VIRT_LLM_QUEUE_LEN;
	struct virt_llm_desc *desc = &vdev->queue[idx];
	__le32 *input = (__le32 *)vdev->input;
	__le32 *kernel = (__le32 *)vdev->input_b;
	__le32 *out = (__le32 *)vdev->output;
	u32 input_values[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	u32 kernel_values[] = { 1, 0, 0, 1 };
	u32 expected[] = { 6, 8, 12, 14 };
	u64 in_dims = 3 | (3ULL << 16) | (2ULL << 32) | (2ULL << 48);
	u64 out_dims = 2 | (2ULL << 16);
	u32 expected_sum = 0;
	int ret;

	for (int i = 0; i < ARRAY_SIZE(input_values); i++)
		input[i] = cpu_to_le32(input_values[i]);
	for (int i = 0; i < ARRAY_SIZE(kernel_values); i++)
		kernel[i] = cpu_to_le32(kernel_values[i]);
	memset(vdev->output, 0, VIRT_LLM_TEST_LEN);
	for (int i = 0; i < ARRAY_SIZE(expected); i++)
		expected_sum += expected[i];

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_CONV2D_U32);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->rsvd0 = cpu_to_le32(9);
	desc->rsvd1 = cpu_to_le64(vdev->input_b_dma);
	desc->rsvd2 = cpu_to_le64(in_dims);
	desc->rsvd3 = cpu_to_le64(out_dims);

	ret = virt_llm_submit_tail(vdev, head + 1, "conv2d self-test");
	if (ret)
		return ret;

	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE ||
	    le32_to_cpu(desc->result) != expected_sum) {
		dev_err(&vdev->pdev->dev,
			"conv2d bad status=0x%08x result=0x%08x expected=0x%08x\n",
			le32_to_cpu(desc->status), le32_to_cpu(desc->result),
			expected_sum);
		return -EIO;
	}

	for (int i = 0; i < ARRAY_SIZE(expected); i++) {
		if (le32_to_cpu(out[i]) != expected[i]) {
			dev_err(&vdev->pdev->dev,
				"conv2d mismatch at %d got=%u expected=%u\n",
				i, le32_to_cpu(out[i]), expected[i]);
			return -EIO;
		}
	}
	ret = virt_llm_check_cq(vdev, 9, VIRT_LLM_OP_CONV2D_U32,
				VIRT_LLM_BACKEND_TENSOR,
				VIRT_LLM_DESC_COMPLETE, expected_sum);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev,
		 "conv2d ok: input=3x3 kernel=2x2 output=2x2 checksum=0x%08x\n",
		 expected_sum);
	return 0;
}

static int virt_llm_run_attention_selftest(struct virt_llm_dev *vdev)
{
	u32 head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	u32 idx = head % VIRT_LLM_QUEUE_LEN;
	struct virt_llm_desc *desc = &vdev->queue[idx];
	__le32 *q = (__le32 *)vdev->input;
	__le32 *k = (__le32 *)vdev->input_b;
	__le32 *v = (__le32 *)vdev->input_c;
	__le32 *out = (__le32 *)vdev->output;
	u32 seq_len = 3;
	u32 head_dim = 2;
	u32 dims = seq_len | (head_dim << 16);
	u32 q_values[] = {
		131072, 0,
		0, 131072,
		131072, 131072,
	};
	u32 k_values[] = {
		131072, 0,
		0, 131072,
		131072, 131072,
	};
	u32 v_values[] = {
		65536, 0,
		0, 65536,
		65536, 65536,
	};
	u32 expected[] = {
		65536, 0,
		1179, 64356,
		64376, 64376,
	};
	u32 expected_sum = 0;
	int ret;

	memset(vdev->input, 0, VIRT_LLM_TEST_LEN);
	memset(vdev->input_b, 0, VIRT_LLM_TEST_LEN);
	memset(vdev->input_c, 0, VIRT_LLM_TEST_LEN);
	memset(vdev->output, 0, VIRT_LLM_TEST_LEN);
	for (int i = 0; i < ARRAY_SIZE(q_values); i++)
		q[i] = cpu_to_le32(q_values[i]);
	for (int i = 0; i < ARRAY_SIZE(k_values); i++)
		k[i] = cpu_to_le32(k_values[i]);
	for (int i = 0; i < ARRAY_SIZE(v_values); i++)
		v[i] = cpu_to_le32(v_values[i]);
	for (int i = 0; i < ARRAY_SIZE(expected); i++)
		expected_sum += expected[i];

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_ATTENTION_Q16);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->len = cpu_to_le32(dims);
	desc->rsvd0 = cpu_to_le32(12);
	desc->rsvd1 = cpu_to_le64(vdev->input_b_dma);
	desc->rsvd2 = cpu_to_le64(vdev->input_c_dma);
	desc->rsvd3 = cpu_to_le64(VIRT_LLM_ATTENTION_CAUSAL);

	ret = virt_llm_submit_tail(vdev, head + 1, "attention self-test");
	if (ret)
		return ret;

	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE ||
	    le32_to_cpu(desc->result) != expected_sum) {
		dev_err(&vdev->pdev->dev,
			"attention bad status=0x%08x result=0x%08x expected=0x%08x\n",
			le32_to_cpu(desc->status), le32_to_cpu(desc->result),
			expected_sum);
		return -EIO;
	}

	for (int i = 0; i < ARRAY_SIZE(expected); i++) {
		if (le32_to_cpu(out[i]) != expected[i]) {
			dev_err(&vdev->pdev->dev,
				"attention mismatch at %d got=%u expected=%u\n",
				i, le32_to_cpu(out[i]), expected[i]);
			return -EIO;
		}
	}

	ret = virt_llm_check_cq(vdev, 12, VIRT_LLM_OP_ATTENTION_Q16,
				VIRT_LLM_BACKEND_TENSOR,
				VIRT_LLM_DESC_COMPLETE, expected_sum);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev,
		 "attention q16 ok: seq=%u head_dim=%u checksum=0x%08x\n",
		 seq_len, head_dim, expected_sum);
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
	desc->rsvd3 = cpu_to_le64(VIRT_LLM_KERNEL_SOFTMAX_Q16);

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
				VIRT_LLM_BACKEND_SCALAR,
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
	desc->rsvd3 = cpu_to_le64(VIRT_LLM_KERNEL_POOL_MAX_U32);

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
				VIRT_LLM_BACKEND_SCALAR,
				VIRT_LLM_DESC_COMPLETE, expected_sum);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev, "pool max ok: count=%zu window=2 checksum=0x%08x\n",
		 ARRAY_SIZE(values), expected_sum);
	return 0;
}

static int virt_llm_run_batch_wrap_selftest(struct virt_llm_dev *vdev)
{
	u32 head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	u32 cq_tail = ioread32(vdev->bar + VIRT_LLM_REG_CQ_TAIL);
	u32 expected_sum[4] = { 0 };
	int ret;

	for (int d = 0; d < 4; d++) {
		u32 idx = (head + d) % VIRT_LLM_QUEUE_LEN;
		struct virt_llm_desc *desc = &vdev->queue[idx];
		u32 offset = d * 16;

		for (int i = 0; i < 8; i++) {
			u8 val = 0x20 + d * 8 + i;

			vdev->input[offset + i] = val;
			vdev->output[offset + i] = 0;
			expected_sum[d] += val;
		}

		memset(desc, 0, sizeof(*desc));
		desc->opcode = cpu_to_le32(VIRT_LLM_OP_DMA_COPY);
		desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
		desc->input_addr = cpu_to_le64(vdev->input_dma + offset);
		desc->output_addr = cpu_to_le64(vdev->output_dma + offset);
		desc->len = cpu_to_le32(8);
		desc->rsvd0 = cpu_to_le32(20 + d);
	}

	ret = virt_llm_submit_tail(vdev, head + 4, "batch wrap self-test");
	if (ret)
		return ret;

	for (int d = 0; d < 4; d++) {
		u32 idx = (head + d) % VIRT_LLM_QUEUE_LEN;
		struct virt_llm_desc *desc = &vdev->queue[idx];
		struct virt_llm_cpl *cpl = &vdev->cq[(cq_tail + d) % VIRT_LLM_CQ_LEN];
		u32 offset = d * 16;

		if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_COMPLETE ||
		    le32_to_cpu(desc->result) != expected_sum[d]) {
			dev_err(&vdev->pdev->dev,
				"batch desc %d bad status=0x%08x result=0x%08x expected=0x%08x\n",
				d, le32_to_cpu(desc->status),
				le32_to_cpu(desc->result), expected_sum[d]);
			return -EIO;
		}

		for (int i = 0; i < 8; i++) {
			if (vdev->output[offset + i] != vdev->input[offset + i]) {
				dev_err(&vdev->pdev->dev,
					"batch copy mismatch desc=%d byte=%d got=0x%02x expected=0x%02x\n",
					d, i, vdev->output[offset + i],
					vdev->input[offset + i]);
				return -EIO;
			}
		}

		if (le32_to_cpu(cpl->command_id) != 20 + d ||
		    le32_to_cpu(cpl->opcode) != VIRT_LLM_OP_DMA_COPY ||
		    le32_to_cpu(cpl->backend) != VIRT_LLM_BACKEND_DMA ||
		    le32_to_cpu(cpl->status) != VIRT_LLM_DESC_COMPLETE ||
		    le32_to_cpu(cpl->result) != expected_sum[d]) {
			dev_err(&vdev->pdev->dev,
				"batch cq %d bad cmd=%u opcode=0x%x backend=%u status=0x%x result=0x%x\n",
				d, le32_to_cpu(cpl->command_id),
				le32_to_cpu(cpl->opcode),
				le32_to_cpu(cpl->backend),
				le32_to_cpu(cpl->status),
				le32_to_cpu(cpl->result));
			return -EIO;
		}
	}

	iowrite32(cq_tail + 4, vdev->bar + VIRT_LLM_REG_CQ_HEAD);
	dev_info(&vdev->pdev->dev,
		 "batch wrap ok: q_head=%u->%u cq_tail=%u->%u descriptors=4\n",
		 head, ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD),
		 cq_tail, ioread32(vdev->bar + VIRT_LLM_REG_CQ_TAIL));
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

static int virt_llm_run_bad_kernel_selftest(struct virt_llm_dev *vdev,
					   u32 kernel_selector,
					   u32 command_id,
					   const char *label)
{
	u32 head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	u32 idx = head % VIRT_LLM_QUEUE_LEN;
	struct virt_llm_desc *desc = &vdev->queue[idx];
	__le32 *a = (__le32 *)vdev->input;
	__le32 *b = (__le32 *)vdev->input_b;
	u32 q_error;
	u32 q_status;
	int ret;

	for (int i = 0; i < 4; i++) {
		a[i] = cpu_to_le32(i + 1);
		b[i] = cpu_to_le32(10 + i);
	}
	memset(vdev->output, 0, VIRT_LLM_TEST_LEN);

	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(VIRT_LLM_OP_VEC_ADD_U32);
	desc->flags = cpu_to_le32(VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(vdev->input_dma);
	desc->output_addr = cpu_to_le64(vdev->output_dma);
	desc->len = cpu_to_le32(4);
	desc->rsvd0 = cpu_to_le32(command_id);
	desc->rsvd1 = cpu_to_le64(vdev->input_b_dma);
	desc->rsvd3 = cpu_to_le64(kernel_selector);

	ret = virt_llm_submit_tail(vdev, head + 1, label);
	if (ret)
		return ret;

	q_status = ioread32(vdev->bar + VIRT_LLM_REG_Q_STATUS);
	q_error = ioread32(vdev->bar + VIRT_LLM_REG_Q_ERROR);
	if (le32_to_cpu(desc->status) != VIRT_LLM_DESC_BAD_KERNEL ||
	    !(q_status & VIRT_LLM_Q_STATUS_ERR) ||
	    q_error != VIRT_LLM_Q_ERR_KERNEL) {
		dev_err(&vdev->pdev->dev,
			"%s bad status desc=0x%08x q_status=0x%08x q_error=%u\n",
			label, le32_to_cpu(desc->status), q_status, q_error);
		return -EIO;
	}
	ret = virt_llm_check_cq(vdev, command_id, VIRT_LLM_OP_VEC_ADD_U32,
				VIRT_LLM_BACKEND_SCALAR,
				VIRT_LLM_DESC_BAD_KERNEL, 0);
	if (ret)
		return ret;

	dev_info(&vdev->pdev->dev,
		 "%s ok: desc_status=0x%08x q_error=%u\n",
		 label, le32_to_cpu(desc->status), q_error);
	return 0;
}

static int virt_llm_run_kernel_table_selftest(struct virt_llm_dev *vdev,
					     u32 scalar_kernels)
{
	if (scalar_kernels != ARRAY_SIZE(virt_llm_expected_kernels)) {
		dev_err(&vdev->pdev->dev,
			"kernel count mismatch got=%u expected=%zu\n",
			scalar_kernels, ARRAY_SIZE(virt_llm_expected_kernels));
		return -EIO;
	}

	for (u32 i = 0; i < ARRAY_SIZE(virt_llm_expected_kernels); i++) {
		const struct virt_llm_kernel_expected *expected;
		u32 kernel_id;
		u32 opcode;
		u32 abi;
		u32 entry;
		u32 size;
		u32 checksum;

		expected = &virt_llm_expected_kernels[i];
		iowrite32(i, vdev->bar + VIRT_LLM_REG_KERNEL_INDEX);
		kernel_id = ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_ID);
		opcode = ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_OPCODE);
		abi = ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_ABI);
		entry = ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_ENTRY);
		size = ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_SIZE);
		checksum = ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_CHECKSUM);

		if (kernel_id != expected->kernel_id ||
		    opcode != expected->opcode ||
		    abi != expected->abi ||
		    entry != expected->entry ||
		    size != expected->size ||
		    checksum != expected->checksum) {
			dev_err(&vdev->pdev->dev,
				"kernel table[%u] mismatch id=%u opcode=0x%x abi=%u entry=0x%x size=%u checksum=0x%x\n",
				i, kernel_id, opcode, abi, entry, size,
				checksum);
			return -EIO;
		}
	}

	iowrite32(scalar_kernels, vdev->bar + VIRT_LLM_REG_KERNEL_INDEX);
	if (ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_ID) ||
	    ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_OPCODE) ||
	    ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_ABI) ||
	    ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_ENTRY) ||
	    ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_SIZE) ||
	    ioread32(vdev->bar + VIRT_LLM_REG_KERNEL_CHECKSUM)) {
		dev_err(&vdev->pdev->dev,
			"out-of-range kernel metadata did not read as zero\n");
		return -EIO;
	}
	iowrite32(0, vdev->bar + VIRT_LLM_REG_KERNEL_INDEX);

	dev_info(&vdev->pdev->dev,
		 "kernel table ok: kernels=%u abi=%u first_entry=0x%08x last_entry=0x%08x\n",
		 scalar_kernels, VIRT_LLM_KERNEL_ABI_VERSION,
		 virt_llm_expected_kernels[0].entry,
		 virt_llm_expected_kernels[ARRAY_SIZE(virt_llm_expected_kernels) - 1].entry);
	return 0;
}

static struct virt_llm_user_dma *
virt_llm_user_buf_get(struct virt_llm_dev *vdev, u32 handle)
{
	if (!handle || handle > VIRT_LLM_USER_BUF_MAX)
		return NULL;
	if (!vdev->user_bufs[handle - 1].in_use)
		return NULL;
	return &vdev->user_bufs[handle - 1];
}

static long virt_llm_ioctl_get_info(struct virt_llm_dev *vdev, void __user *argp)
{
	struct virt_llm_user_info info = {
		.magic = ioread32(vdev->bar + VIRT_LLM_REG_MAGIC),
		.version = ioread32(vdev->bar + VIRT_LLM_REG_VERSION),
		.features = ioread32(vdev->bar + VIRT_LLM_REG_FEATURES),
		.abi = ioread32(vdev->bar + VIRT_LLM_REG_ABI),
		.q_max = ioread32(vdev->bar + VIRT_LLM_REG_Q_MAX),
		.xfer_max = ioread32(vdev->bar + VIRT_LLM_REG_XFER_MAX),
		.cq_size = VIRT_LLM_CQ_LEN,
	};

	if (copy_to_user(argp, &info, sizeof(info)))
		return -EFAULT;
	return 0;
}

static long virt_llm_ioctl_alloc_buffer(struct virt_llm_dev *vdev,
					void __user *argp)
{
	struct virt_llm_user_buffer req;
	struct virt_llm_user_dma *buf = NULL;
	struct device *dev = &vdev->pdev->dev;
	int i;

	if (copy_from_user(&req, argp, sizeof(req)))
		return -EFAULT;
	if (!req.size || req.size > VIRT_LLM_USER_BUF_SIZE)
		return -EINVAL;

	mutex_lock(&vdev->user_lock);
	for (i = 0; i < VIRT_LLM_USER_BUF_MAX; i++) {
		if (!vdev->user_bufs[i].in_use) {
			buf = &vdev->user_bufs[i];
			break;
		}
	}
	if (!buf) {
		mutex_unlock(&vdev->user_lock);
		return -ENOSPC;
	}

	buf->cpu = dma_alloc_coherent(dev, VIRT_LLM_USER_BUF_SIZE, &buf->dma,
				      GFP_KERNEL);
	if (!buf->cpu) {
		mutex_unlock(&vdev->user_lock);
		return -ENOMEM;
	}
	buf->size = VIRT_LLM_USER_BUF_SIZE;
	buf->in_use = true;
	req.handle = i + 1;
	req.size = buf->size;
	req.dma_addr = buf->dma;
	mutex_unlock(&vdev->user_lock);

	if (copy_to_user(argp, &req, sizeof(req)))
		return -EFAULT;
	return 0;
}

static long virt_llm_ioctl_free_buffer(struct virt_llm_dev *vdev,
				       void __user *argp)
{
	struct device *dev = &vdev->pdev->dev;
	struct virt_llm_user_dma *buf;
	u32 handle;

	if (copy_from_user(&handle, argp, sizeof(handle)))
		return -EFAULT;

	mutex_lock(&vdev->user_lock);
	buf = virt_llm_user_buf_get(vdev, handle);
	if (!buf) {
		mutex_unlock(&vdev->user_lock);
		return -ENOENT;
	}
	dma_free_coherent(dev, buf->size, buf->cpu, buf->dma);
	memset(buf, 0, sizeof(*buf));
	mutex_unlock(&vdev->user_lock);
	return 0;
}

static long virt_llm_ioctl_submit_desc(struct virt_llm_dev *vdev,
				       void __user *argp)
{
	struct virt_llm_user_desc req;
	struct virt_llm_user_dma *input;
	struct virt_llm_user_dma *output;
	struct virt_llm_user_dma *rsvd1 = NULL;
	struct virt_llm_user_dma *rsvd2 = NULL;
	struct virt_llm_desc *desc;
	u32 head;
	u32 tail;

	if (copy_from_user(&req, argp, sizeof(req)))
		return -EFAULT;

	mutex_lock(&vdev->user_lock);
	input = virt_llm_user_buf_get(vdev, req.input_handle);
	output = virt_llm_user_buf_get(vdev, req.output_handle);
	if (req.rsvd1_handle)
		rsvd1 = virt_llm_user_buf_get(vdev, req.rsvd1_handle);
	if (req.rsvd2_handle)
		rsvd2 = virt_llm_user_buf_get(vdev, req.rsvd2_handle);
	if (!input || !output || (req.rsvd1_handle && !rsvd1) ||
	    (req.rsvd2_handle && !rsvd2)) {
		mutex_unlock(&vdev->user_lock);
		return -EINVAL;
	}

	head = ioread32(vdev->bar + VIRT_LLM_REG_Q_HEAD);
	tail = head + 1;
	desc = &vdev->queue[head % VIRT_LLM_QUEUE_LEN];
	memset(desc, 0, sizeof(*desc));
	desc->opcode = cpu_to_le32(req.opcode);
	desc->flags = cpu_to_le32(req.flags | VIRT_LLM_DESC_F_READY);
	desc->input_addr = cpu_to_le64(input->dma);
	desc->output_addr = cpu_to_le64(output->dma);
	desc->len = cpu_to_le32(req.len);
	if (!req.command_id)
		req.command_id = ++vdev->next_command_id;
	desc->rsvd0 = cpu_to_le32(req.command_id);
	desc->rsvd1 = cpu_to_le64(rsvd1 ? rsvd1->dma : req.rsvd1_addr);
	desc->rsvd2 = cpu_to_le64(rsvd2 ? rsvd2->dma : req.rsvd2_addr);
	desc->rsvd3 = cpu_to_le64(req.rsvd3);

	reinit_completion(&vdev->done);
	dma_wmb();
	iowrite32(tail, vdev->bar + VIRT_LLM_REG_Q_TAIL);
	iowrite32(VIRT_LLM_CMD_KICK, vdev->bar + VIRT_LLM_REG_COMMAND);
	mutex_unlock(&vdev->user_lock);

	if (copy_to_user(argp, &req, sizeof(req)))
		return -EFAULT;
	return 0;
}

static long virt_llm_ioctl_wait_cq(struct virt_llm_dev *vdev, void __user *argp)
{
	struct virt_llm_user_cpl out;
	struct virt_llm_cpl *cpl;
	unsigned long timeout;
	u32 tail;

	timeout = wait_for_completion_timeout(&vdev->done, msecs_to_jiffies(5000));
	if (!timeout)
		return -ETIMEDOUT;

	mutex_lock(&vdev->user_lock);
	dma_rmb();
	tail = ioread32(vdev->bar + VIRT_LLM_REG_CQ_TAIL);
	if (vdev->user_cq_head == tail) {
		mutex_unlock(&vdev->user_lock);
		return -EAGAIN;
	}

	cpl = &vdev->cq[vdev->user_cq_head % VIRT_LLM_CQ_LEN];
	out.command_id = le32_to_cpu(cpl->command_id);
	out.opcode = le32_to_cpu(cpl->opcode);
	out.backend = le32_to_cpu(cpl->backend);
	out.status = le32_to_cpu(cpl->status);
	out.result = le32_to_cpu(cpl->result);
	out.q_head = le32_to_cpu(cpl->q_head);
	out.q_error = ioread32(vdev->bar + VIRT_LLM_REG_Q_ERROR);
	out.reserved = 0;

	vdev->user_cq_head++;
	iowrite32(vdev->user_cq_head, vdev->bar + VIRT_LLM_REG_CQ_HEAD);
	mutex_unlock(&vdev->user_lock);

	if (copy_to_user(argp, &out, sizeof(out)))
		return -EFAULT;
	return 0;
}

static long virt_llm_unlocked_ioctl(struct file *file, unsigned int cmd,
				    unsigned long arg)
{
	struct virt_llm_dev *vdev = file->private_data;
	void __user *argp = (void __user *)arg;

	switch (cmd) {
	case VIRT_LLM_IOCTL_GET_INFO:
		return virt_llm_ioctl_get_info(vdev, argp);
	case VIRT_LLM_IOCTL_ALLOC_BUFFER:
		return virt_llm_ioctl_alloc_buffer(vdev, argp);
	case VIRT_LLM_IOCTL_FREE_BUFFER:
		return virt_llm_ioctl_free_buffer(vdev, argp);
	case VIRT_LLM_IOCTL_SUBMIT_DESC:
		return virt_llm_ioctl_submit_desc(vdev, argp);
	case VIRT_LLM_IOCTL_WAIT_CQ:
		return virt_llm_ioctl_wait_cq(vdev, argp);
	default:
		return -ENOTTY;
	}
}

static int virt_llm_open(struct inode *inode, struct file *file)
{
	if (!virt_llm_singleton)
		return -ENODEV;
	file->private_data = virt_llm_singleton;
	return 0;
}

static int virt_llm_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct virt_llm_dev *vdev = file->private_data;
	u32 handle = vma->vm_pgoff + 1;
	struct virt_llm_user_dma *buf;
	unsigned long size = vma->vm_end - vma->vm_start;
	int ret;

	mutex_lock(&vdev->user_lock);
	buf = virt_llm_user_buf_get(vdev, handle);
	if (!buf || size > buf->size) {
		mutex_unlock(&vdev->user_lock);
		return -EINVAL;
	}
	vma->vm_pgoff = 0;
	ret = dma_mmap_coherent(&vdev->pdev->dev, vma, buf->cpu, buf->dma,
				buf->size);
	if (ret)
		dev_err(&vdev->pdev->dev,
			"user mmap failed handle=%u size=%lu buf_size=%zu dma=%pad ret=%d\n",
			handle, size, buf->size, &buf->dma, ret);
	mutex_unlock(&vdev->user_lock);
	return ret;
}

static const struct file_operations virt_llm_fops = {
	.owner = THIS_MODULE,
	.open = virt_llm_open,
	.unlocked_ioctl = virt_llm_unlocked_ioctl,
	.mmap = virt_llm_mmap,
	.llseek = noop_llseek,
};

static int virt_llm_register_misc(struct virt_llm_dev *vdev)
{
	memset(vdev->queue, 0, sizeof(*vdev->queue) * VIRT_LLM_QUEUE_LEN);
	memset(vdev->cq, 0, sizeof(*vdev->cq) * VIRT_LLM_CQ_LEN);
	iowrite32(VIRT_LLM_Q_CTRL_RESET, vdev->bar + VIRT_LLM_REG_Q_CTRL);
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

	vdev->miscdev.minor = VIRT_LLM_MISC_MINOR;
	vdev->miscdev.name = "virt_llm0";
	vdev->miscdev.fops = &virt_llm_fops;
	vdev->miscdev.parent = &vdev->pdev->dev;
	vdev->miscdev.mode = 0600;
	vdev->user_cq_head = 0;
	vdev->next_command_id = 1000;
	virt_llm_singleton = vdev;
	return misc_register(&vdev->miscdev);
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
	u32 scalar_kernels;
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
	scalar_kernels = ioread32(bar + VIRT_LLM_REG_SCALAR_KERNELS);
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
	mutex_init(&vdev->user_lock);
	pci_set_drvdata(pdev, vdev);

	ret = virt_llm_alloc_dma(vdev);
	if (ret)
		return ret;
	dev_info(&pdev->dev, "dma buffers: queue=%pad cq=%pad input=%pad output=%pad\n",
		 &vdev->queue_dma, &vdev->cq_dma, &vdev->input_dma,
		 &vdev->output_dma);
	dev_info(&pdev->dev, "dma input_b=%pad input_c=%pad\n",
		 &vdev->input_b_dma, &vdev->input_c_dma);

	ret = virt_llm_request_irq(vdev);
	if (ret)
		return ret;

	if (run_selftest) {
		ret = virt_llm_run_kernel_table_selftest(vdev, scalar_kernels);
		if (ret) {
			pci_free_irq_vectors(pdev);
			return ret;
		}

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

		ret = virt_llm_run_dot_selftest(vdev);
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

		ret = virt_llm_run_conv2d_selftest(vdev);
		if (ret) {
			pci_free_irq_vectors(pdev);
			return ret;
		}

		ret = virt_llm_run_attention_selftest(vdev);
		if (ret) {
			pci_free_irq_vectors(pdev);
			return ret;
		}

		ret = virt_llm_run_batch_wrap_selftest(vdev);
		if (ret) {
			pci_free_irq_vectors(pdev);
			return ret;
		}

		ret = virt_llm_run_bad_kernel_selftest(vdev, 0x00ff, 10,
						       "bad kernel id self-test");
		if (ret) {
			pci_free_irq_vectors(pdev);
			return ret;
		}

		ret = virt_llm_run_bad_kernel_selftest(vdev,
						       VIRT_LLM_KERNEL_VEC_ADD_U32 |
						       (2U << 16), 11,
						       "bad kernel abi self-test");
		if (ret) {
			pci_free_irq_vectors(pdev);
			return ret;
		}

		ret = virt_llm_run_error_selftest(vdev);
		if (ret) {
			pci_free_irq_vectors(pdev);
			return ret;
		}
	} else {
		iowrite32(lower_32_bits(vdev->queue_dma), vdev->bar + VIRT_LLM_REG_Q_LO);
		iowrite32(upper_32_bits(vdev->queue_dma), vdev->bar + VIRT_LLM_REG_Q_HI);
		iowrite32(VIRT_LLM_QUEUE_LEN, vdev->bar + VIRT_LLM_REG_Q_SIZE);
		iowrite32(lower_32_bits(vdev->cq_dma), vdev->bar + VIRT_LLM_REG_CQ_LO);
		iowrite32(upper_32_bits(vdev->cq_dma), vdev->bar + VIRT_LLM_REG_CQ_HI);
		iowrite32(VIRT_LLM_CQ_LEN, vdev->bar + VIRT_LLM_REG_CQ_SIZE);
		iowrite32(VIRT_LLM_IRQ_ALL, vdev->bar + VIRT_LLM_REG_IRQ_MASK);
		iowrite32(VIRT_LLM_Q_CTRL_ENABLE, vdev->bar + VIRT_LLM_REG_Q_CTRL);
		if (!(ioread32(vdev->bar + VIRT_LLM_REG_Q_STATUS) & VIRT_LLM_Q_STATUS_EN)) {
			pci_free_irq_vectors(pdev);
			return -EIO;
		}
	}

	ret = virt_llm_register_misc(vdev);
	if (ret) {
		pci_free_irq_vectors(pdev);
		return ret;
	}

	dev_info(&pdev->dev,
		 "probe ok: magic=0x%08x version=%u abi=%u q_max=%u xfer_max=%u irq_vec=%u scalar_kernels=%u last_kernel=%u last_opcode=0x%08x doorbell=0x%08x status=0x%08x\n",
		 magic, version, abi, q_max, xfer_max, irq_vec, scalar_kernels,
		 ioread32(bar + VIRT_LLM_REG_SCALAR_LAST_KERNEL),
		 ioread32(bar + VIRT_LLM_REG_SCALAR_LAST_OPCODE),
		 VIRT_LLM_TEST_DOORBELL, status);

	return 0;
}

static void virt_llm_pci_remove(struct pci_dev *pdev)
{
	struct virt_llm_dev *vdev = pci_get_drvdata(pdev);
	struct device *dev = &pdev->dev;

	if (vdev) {
		misc_deregister(&vdev->miscdev);
		if (virt_llm_singleton == vdev)
			virt_llm_singleton = NULL;
		for (int i = 0; i < VIRT_LLM_USER_BUF_MAX; i++) {
			if (vdev->user_bufs[i].in_use)
				dma_free_coherent(dev, vdev->user_bufs[i].size,
						  vdev->user_bufs[i].cpu,
						  vdev->user_bufs[i].dma);
		}
	}
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
