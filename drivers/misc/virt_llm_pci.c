// SPDX-License-Identifier: GPL-2.0-only
/*
 * Minimal PCI driver for the QEMU virt-llm prototype device.
 */

#include <linux/io.h>
#include <linux/module.h>
#include <linux/pci.h>

#define VIRT_LLM_DEVICE_ID      0x1100

#define VIRT_LLM_REG_MAGIC      0x00
#define VIRT_LLM_REG_VERSION    0x04
#define VIRT_LLM_REG_DOORBELL   0x08
#define VIRT_LLM_REG_STATUS     0x0c

#define VIRT_LLM_MAGIC          0x4c4c4d31u /* "LLM1" */
#define VIRT_LLM_STATUS_XOR     0xa5a5a5a5u
#define VIRT_LLM_TEST_DOORBELL  0x13579bdfu

static int virt_llm_pci_probe(struct pci_dev *pdev,
			      const struct pci_device_id *id)
{
	void __iomem *bar;
	u32 magic;
	u32 version;
	u32 status;
	u32 expected;
	int ret;

	ret = pcim_enable_device(pdev);
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

	dev_info(&pdev->dev,
		 "probe ok: magic=0x%08x version=%u doorbell=0x%08x status=0x%08x\n",
		 magic, version, VIRT_LLM_TEST_DOORBELL, status);

	return 0;
}

static void virt_llm_pci_remove(struct pci_dev *pdev)
{
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
