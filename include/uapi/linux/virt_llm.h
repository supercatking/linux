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
#define VIRT_LLM_OP_MODEL_LOAD     0x0300
#define VIRT_LLM_OP_MODEL_QUERY    0x0301
#define VIRT_LLM_OP_EMBED_LOOKUP_F32 0x0310
#define VIRT_LLM_OP_RMSNORM_F32    0x0311
#define VIRT_LLM_OP_ROPE_F32       0x0312
#define VIRT_LLM_OP_GEMM_F32       0x0313
#define VIRT_LLM_OP_ADD_F32        0x0314
#define VIRT_LLM_OP_SWIGLU_F32     0x0315
#define VIRT_LLM_OP_QWEN_GQA_ATTENTION_F32 0x0316
#define VIRT_LLM_OP_LM_HEAD_F32    0x0317
#define VIRT_LLM_OP_ARGMAX_F32     0x0318

#define VIRT_LLM_DESC_F_READY      (1U << 0)
#define VIRT_LLM_DESC_COMPLETE     1
#define VIRT_LLM_DESC_UNSUPP       0x80000002U
#define VIRT_LLM_DESC_BAD_KERNEL   0x80000004U
#define VIRT_LLM_DESC_BAD_TENSOR   0x80000005U

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

#define VIRT_LLM_TENSOR_ABI_VERSION 1
#define VIRT_LLM_DTYPE_U32          1
#define VIRT_LLM_DTYPE_F32          2
#define VIRT_LLM_DTYPE_BF16         3

#define VIRT_LLM_QWEN_LAYERS        24
#define VIRT_LLM_QWEN_HIDDEN        896
#define VIRT_LLM_QWEN_HEADS         14
#define VIRT_LLM_QWEN_KV_HEADS      2
#define VIRT_LLM_QWEN_HEAD_DIM      64
#define VIRT_LLM_QWEN_INTERMEDIATE  4864
#define VIRT_LLM_QWEN_VOCAB         151936

#define VIRT_LLM_TENSOR_F_CAUSAL    (1U << 0)

#define VIRT_LLM_TENSOR_QWEN_EMBED      1
#define VIRT_LLM_TENSOR_QWEN_FINAL_NORM 2
#define VIRT_LLM_TENSOR_QWEN_LAYER_BASE 1000
#define VIRT_LLM_TENSOR_QWEN_LAYER_STRIDE 16
#define VIRT_LLM_TENSOR_QWEN_INPUT_NORM 0
#define VIRT_LLM_TENSOR_QWEN_POST_NORM  1
#define VIRT_LLM_TENSOR_QWEN_Q_PROJ     2
#define VIRT_LLM_TENSOR_QWEN_K_PROJ     3
#define VIRT_LLM_TENSOR_QWEN_V_PROJ     4
#define VIRT_LLM_TENSOR_QWEN_O_PROJ     5
#define VIRT_LLM_TENSOR_QWEN_GATE_PROJ  6
#define VIRT_LLM_TENSOR_QWEN_UP_PROJ    7
#define VIRT_LLM_TENSOR_QWEN_DOWN_PROJ  8
#define VIRT_LLM_TENSOR_QWEN_Q_BIAS     9
#define VIRT_LLM_TENSOR_QWEN_K_BIAS     10
#define VIRT_LLM_TENSOR_QWEN_V_BIAS     11

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

struct virt_llm_tensor_req {
	__u32 abi;
	__u32 dtype;
	__u32 rank;
	__u32 flags;
	__u32 layer_id;
	__u32 tensor_id;
	__u32 aux_tensor_id;
	__u32 reserved0;
	__u32 dims[4];
	__u32 input_offset;
	__u32 weight_offset;
	__u32 aux_offset;
	__u32 output_offset;
	__u32 input2_offset;
	__u32 reserved1;
	__u64 scalar0_bits;
	__u64 scalar1_bits;
};

struct virt_llm_model_query {
	__u32 abi;
	__u32 model_loaded;
	__u32 layers;
	__u32 hidden_size;
	__u32 attention_heads;
	__u32 kv_heads;
	__u32 head_dim;
	__u32 intermediate_size;
	__u32 vocab_size;
	__u32 dtype;
	__u64 rope_theta_bits;
	__u64 rms_eps_bits;
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
