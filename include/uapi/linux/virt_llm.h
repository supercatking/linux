/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef _UAPI_LINUX_VIRT_LLM_H
#define _UAPI_LINUX_VIRT_LLM_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define VIRT_LLM_IOCTL_MAGIC 'L'

#define VIRT_LLM_OP_INFER          0x0001
#define VIRT_LLM_OP_DMA_COPY       0x0010
#define VIRT_LLM_OP_VEC_ADD_U32    0x0100
#define VIRT_LLM_OP_SOFTMAX_Q16    0x0101
#define VIRT_LLM_OP_POOL_MAX_U32   0x0102
#define VIRT_LLM_OP_DOT_U32        0x0103
#define VIRT_LLM_OP_GEMM_U32       0x0200
#define VIRT_LLM_OP_CONV2D_U32     0x0201
#define VIRT_LLM_OP_ATTENTION_Q16  0x0202

#define VIRT_LLM_DESC_F_READY      (1U << 0)
#define VIRT_LLM_DESC_COMPLETE     1
#define VIRT_LLM_DESC_UNSUPP       0x80000002U
#define VIRT_LLM_DESC_BAD_KERNEL   0x80000004U

#define VIRT_LLM_BACKEND_COMPAT    0
#define VIRT_LLM_BACKEND_DMA       1
#define VIRT_LLM_BACKEND_VECTOR    2
#define VIRT_LLM_BACKEND_TENSOR    3
#define VIRT_LLM_BACKEND_SCALAR    4

#define VIRT_LLM_KERNEL_VEC_ADD_U32  1
#define VIRT_LLM_KERNEL_DOT_U32      2
#define VIRT_LLM_KERNEL_SOFTMAX_Q16  3
#define VIRT_LLM_KERNEL_POOL_MAX_U32 4
#define VIRT_LLM_KERNEL_ABI_VERSION  1

#define VIRT_LLM_ATTENTION_CAUSAL  (1U << 0)

struct virt_llm_user_info {
	__u32 magic;
	__u32 version;
	__u32 features;
	__u32 abi;
	__u32 q_max;
	__u32 xfer_max;
	__u32 cq_size;
	__u32 reserved;
};

struct virt_llm_user_buffer {
	__u32 handle;
	__u32 size;
	__u64 dma_addr;
};

struct virt_llm_user_desc {
	__u32 opcode;
	__u32 flags;
	__u32 input_handle;
	__u32 output_handle;
	__u32 len;
	__u32 command_id;
	__u32 rsvd0;
	__u32 rsvd1_handle;
	__u32 rsvd2_handle;
	__u32 reserved;
	__u64 rsvd1_addr;
	__u64 rsvd2_addr;
	__u64 rsvd3;
};

struct virt_llm_user_cpl {
	__u32 command_id;
	__u32 opcode;
	__u32 backend;
	__u32 status;
	__u32 result;
	__u32 q_head;
	__u32 q_error;
	__u32 reserved;
};

#define VIRT_LLM_IOCTL_GET_INFO \
	_IOR(VIRT_LLM_IOCTL_MAGIC, 0x00, struct virt_llm_user_info)
#define VIRT_LLM_IOCTL_ALLOC_BUFFER \
	_IOWR(VIRT_LLM_IOCTL_MAGIC, 0x01, struct virt_llm_user_buffer)
#define VIRT_LLM_IOCTL_FREE_BUFFER \
	_IOW(VIRT_LLM_IOCTL_MAGIC, 0x02, __u32)
#define VIRT_LLM_IOCTL_SUBMIT_DESC \
	_IOWR(VIRT_LLM_IOCTL_MAGIC, 0x03, struct virt_llm_user_desc)
#define VIRT_LLM_IOCTL_WAIT_CQ \
	_IOR(VIRT_LLM_IOCTL_MAGIC, 0x04, struct virt_llm_user_cpl)

#endif /* _UAPI_LINUX_VIRT_LLM_H */
