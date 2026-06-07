// SPDX-License-Identifier: GPL-2.0-only
/*
 * Freestanding riscv32 /init for a per-op Qwen smoke path.
 */

typedef unsigned int u32;
typedef unsigned long ulong;
typedef unsigned long long u64;

#define AT_FDCWD        -100
#define O_RDONLY        00
#define O_RDWR          02
#define S_IFCHR         0020000
#define PROT_READ       0x1
#define PROT_WRITE      0x2
#define MAP_SHARED      0x01
#define BUF_SIZE        (1024 * 1024)
#define DATA_OFF        4096

#define VIRT_LLM_IOCTL_MAGIC 'L'
#define VIRT_LLM_OP_MODEL_LOAD     0x0300
#define VIRT_LLM_OP_EMBED_LOOKUP_F32 0x0310
#define VIRT_LLM_OP_RMSNORM_F32    0x0311
#define VIRT_LLM_OP_ROPE_F32       0x0312
#define VIRT_LLM_OP_GEMM_F32       0x0313
#define VIRT_LLM_OP_ADD_F32        0x0314
#define VIRT_LLM_OP_SWIGLU_F32     0x0315
#define VIRT_LLM_OP_QWEN_GQA_ATTENTION_F32 0x0316
#define VIRT_LLM_OP_LM_HEAD_F32    0x0317
#define VIRT_LLM_OP_ARGMAX_F32     0x0318
#define VIRT_LLM_DESC_COMPLETE     1
#define VIRT_LLM_BACKEND_DMA       1
#define VIRT_LLM_BACKEND_VECTOR    2
#define VIRT_LLM_BACKEND_TENSOR    3
#define VIRT_LLM_TENSOR_ABI_VERSION 1
#define VIRT_LLM_DTYPE_F32         2
#define VIRT_LLM_TENSOR_F_CAUSAL   (1U << 0)

#define QWEN_HIDDEN        896
#define QWEN_HEADS         14
#define QWEN_KV_HEADS      2
#define QWEN_HEAD_DIM      64
#define QWEN_INTERMEDIATE  4864
#define QWEN_VOCAB         151936
#define QWEN_LAYER_COUNT   24
#ifndef QWEN_FIXTURE_PROMPT
#define QWEN_FIXTURE_PROMPT "What is the capital of France?"
#endif
#ifndef QWEN_PROMPT_TOKEN_COUNT
#define QWEN_PROMPT_TOKEN_COUNT 36
#endif
#ifndef QWEN_PROMPT_TOKENS
#define QWEN_PROMPT_TOKENS \
	{ 151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, \
	  54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, \
	  198, 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, \
	  151645, 198, 151644, 77091, 198 }
#endif
#ifndef QWEN_EXPECTED_FIRST_TOKEN
#define QWEN_EXPECTED_FIRST_TOKEN 785
#endif
#ifndef QWEN_DECODE_STEPS
#define QWEN_DECODE_STEPS 1
#endif
#define QWEN_MAX_CONTEXT   (QWEN_PROMPT_TOKEN_COUNT + QWEN_DECODE_STEPS)

#define T_EMBED       1
#define T_FINAL_NORM  2
#define T_LAYER_BASE  1000
#define T_LAYER_STRIDE 16
#define T_INPUT_NORM 0
#define T_POST_NORM  1
#define T_Q_PROJ     2
#define T_K_PROJ     3
#define T_V_PROJ     4
#define T_O_PROJ     5
#define T_GATE_PROJ  6
#define T_UP_PROJ    7
#define T_DOWN_PROJ  8
#define T_Q_BIAS     9
#define T_K_BIAS     10
#define T_V_BIAS     11

#define _IOC_NRBITS     8
#define _IOC_TYPEBITS   8
#define _IOC_SIZEBITS   14
#define _IOC_NRSHIFT    0
#define _IOC_TYPESHIFT  (_IOC_NRSHIFT + _IOC_NRBITS)
#define _IOC_SIZESHIFT  (_IOC_TYPESHIFT + _IOC_TYPEBITS)
#define _IOC_DIRSHIFT   (_IOC_SIZESHIFT + _IOC_SIZEBITS)
#define _IOC_WRITE      1U
#define _IOC_READ       2U
#define _IOC(dir, type, nr, size) \
	(((dir) << _IOC_DIRSHIFT) | ((type) << _IOC_TYPESHIFT) | \
	 ((nr) << _IOC_NRSHIFT) | ((size) << _IOC_SIZESHIFT))
#define _IOR(type, nr, size) _IOC(_IOC_READ, (type), (nr), sizeof(size))
#define _IOWR(type, nr, size) _IOC(_IOC_READ | _IOC_WRITE, (type), (nr), sizeof(size))

struct virt_llm_user_buffer {
	u32 handle;
	u32 size;
	u64 dma_addr;
};

struct virt_llm_user_desc {
	u32 opcode;
	u32 flags;
	u32 input_handle;
	u32 output_handle;
	u32 len;
	u32 command_id;
	u32 rsvd0;
	u32 rsvd1_handle;
	u32 rsvd2_handle;
	u32 reserved;
	u64 rsvd1_addr;
	u64 rsvd2_addr;
	u64 rsvd3;
};

struct virt_llm_user_cpl {
	u32 command_id;
	u32 opcode;
	u32 backend;
	u32 status;
	u32 result;
	u32 q_head;
	u32 q_error;
	u32 reserved;
};

struct virt_llm_tensor_req {
	u32 abi;
	u32 dtype;
	u32 rank;
	u32 flags;
	u32 layer_id;
	u32 tensor_id;
	u32 aux_tensor_id;
	u32 reserved0;
	u32 dims[4];
	u32 input_offset;
	u32 weight_offset;
	u32 aux_offset;
	u32 output_offset;
	u32 input2_offset;
	u32 reserved1;
	u64 scalar0_bits;
	u64 scalar1_bits;
};

#define VIRT_LLM_IOCTL_ALLOC_BUFFER \
	_IOWR(VIRT_LLM_IOCTL_MAGIC, 0x01, struct virt_llm_user_buffer)
#define VIRT_LLM_IOCTL_SUBMIT_DESC \
	_IOWR(VIRT_LLM_IOCTL_MAGIC, 0x03, struct virt_llm_user_desc)
#define VIRT_LLM_IOCTL_WAIT_CQ \
	_IOR(VIRT_LLM_IOCTL_MAGIC, 0x04, struct virt_llm_user_cpl)

struct buf {
	struct virt_llm_user_buffer ubuf;
	u32 *ptr;
};

static const u32 qwen_prompt_tokens[QWEN_PROMPT_TOKEN_COUNT] = QWEN_PROMPT_TOKENS;

static inline long syscall0(long n)
{
	register long a7 asm("a7") = n;
	register long a0 asm("a0");

	asm volatile ("ecall" : "=r"(a0) : "r"(a7) : "memory");
	return a0;
}

static inline long syscall1(long n, long a)
{
	register long a7 asm("a7") = n;
	register long a0 asm("a0") = a;

	asm volatile ("ecall" : "+r"(a0) : "r"(a7) : "memory");
	return a0;
}

static inline long syscall3(long n, long a, long b, long c)
{
	register long a7 asm("a7") = n;
	register long a0 asm("a0") = a;
	register long a1 asm("a1") = b;
	register long a2 asm("a2") = c;

	asm volatile ("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
	return a0;
}

static inline long syscall4(long n, long a, long b, long c, long d)
{
	register long a7 asm("a7") = n;
	register long a0 asm("a0") = a;
	register long a1 asm("a1") = b;
	register long a2 asm("a2") = c;
	register long a3 asm("a3") = d;

	asm volatile ("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a3), "r"(a7) : "memory");
	return a0;
}

static inline long syscall6(long n, long a, long b, long c, long d, long e, long f)
{
	register long a7 asm("a7") = n;
	register long a0 asm("a0") = a;
	register long a1 asm("a1") = b;
	register long a2 asm("a2") = c;
	register long a3 asm("a3") = d;
	register long a4 asm("a4") = e;
	register long a5 asm("a5") = f;

	asm volatile ("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a3), "r"(a4),
		      "r"(a5), "r"(a7) : "memory");
	return a0;
}

static unsigned int strlen_(const char *s)
{
	unsigned int n = 0;

	while (s[n])
		n++;
	return n;
}

static void puts_(const char *s)
{
	syscall3(64, 1, (long)s, strlen_(s));
}

static void *memset_(void *p, int v, unsigned int n)
{
	unsigned char *b = p;

	for (unsigned int i = 0; i < n; i++)
		b[i] = v;
	return p;
}

void *memset(void *p, int v, unsigned int n)
{
	return memset_(p, v, n);
}

void *memcpy(void *dst, const void *src, unsigned int n)
{
	unsigned char *d = dst;
	const unsigned char *s = src;

	for (unsigned int i = 0; i < n; i++)
		d[i] = s[i];
	return dst;
}

static void print_dec(u32 v)
{
	char buf[12];
	int i = 10;

	buf[11] = 0;
	if (!v) {
		puts_("0");
		return;
	}
	while (v && i >= 0) {
		buf[i--] = '0' + (v % 10);
		v /= 10;
	}
	puts_(&buf[i + 1]);
}

static void print_hex(u32 v)
{
	char out[] = "0x00000000";
	const char h[] = "0123456789abcdef";

	for (int i = 0; i < 8; i++)
		out[2 + i] = h[(v >> (28 - i * 4)) & 0xf];
	puts_(out);
}

static void print_token_list(const u32 *tokens, u32 count)
{
	for (u32 i = 0; i < count; i++) {
		print_dec(tokens[i]);
		if (i + 1 != count)
			puts_(",");
	}
}

static u32 devno(unsigned int major, unsigned int minor)
{
	return ((minor & 0xff) | (major << 8) | ((minor & ~0xff) << 12));
}


static int name_eq(const char *a, const char *b)
{
	while (*a && *b && *a == *b) {
		a++;
		b++;
	}
	return !*a && !*b;
}

static int parse_misc_minor(const char *buf, unsigned int len)
{
	unsigned int i = 0;

	while (i < len) {
		unsigned int minor = 0;
		char name[32];
		unsigned int n = 0;
		int have_digit = 0;

		while (i < len && (buf[i] == ' ' || buf[i] == '\t'))
			i++;
		while (i < len && buf[i] >= '0' && buf[i] <= '9') {
			have_digit = 1;
			minor = minor * 10 + (buf[i] - '0');
			i++;
		}
		while (i < len && (buf[i] == ' ' || buf[i] == '\t'))
			i++;
		while (i < len && buf[i] != '\n' && n + 1 < sizeof(name))
			name[n++] = buf[i++];
		name[n] = 0;
		while (i < len && buf[i] != '\n')
			i++;
		if (i < len && buf[i] == '\n')
			i++;
		if (have_digit && name_eq(name, "virt_llm0"))
			return (int)minor;
	}
	return -1;
}

static int find_virt_llm_minor(void)
{
	char buf[1024];
	long fd;
	long n;

	fd = syscall4(56, AT_FDCWD, (long)"/proc/misc", O_RDONLY, 0);
	if (fd < 0)
		return -1;
	n = syscall3(63, fd, (long)buf, sizeof(buf) - 1);
	syscall1(57, fd);
	if (n <= 0)
		return -1;
	buf[n] = 0;
	return parse_misc_minor(buf, (unsigned int)n);
}

static int open_dev(void)
{
	long fd;
	int minor;
	long rc;

	fd = syscall4(56, AT_FDCWD, (long)"/dev/virt_llm0", O_RDWR, 0);
	if (fd >= 0)
		return fd;

	syscall3(34, AT_FDCWD, (long)"/dev", 0755);
	minor = find_virt_llm_minor();
	if (minor < 0)
		minor = 243;
	rc = syscall4(33, AT_FDCWD, (long)"/dev/virt_llm0", S_IFCHR | 0600,
		      devno(10, (unsigned int)minor));
	if (rc < 0)
		puts_("mknod /dev/virt_llm0 failed\n");
	return syscall4(56, AT_FDCWD, (long)"/dev/virt_llm0", O_RDWR, 0);
}

static int alloc_buffer(int fd, struct buf *b)
{
	ulong addr;

	memset_(b, 0, sizeof(*b));
	b->ubuf.size = BUF_SIZE;
	if (syscall3(29, fd, VIRT_LLM_IOCTL_ALLOC_BUFFER, (long)&b->ubuf) < 0)
		return -1;
	addr = (ulong)syscall6(222, 0, b->ubuf.size, PROT_READ | PROT_WRITE,
			       MAP_SHARED, fd, b->ubuf.handle - 1);
	if (addr >= (ulong)-4095)
		return -1;
	b->ptr = (u32 *)addr;
	return 0;
}

static int wait_cq(int fd, struct virt_llm_user_cpl *cpl)
{
	memset_(cpl, 0, sizeof(*cpl));
	return syscall3(29, fd, VIRT_LLM_IOCTL_WAIT_CQ, (long)cpl) < 0 ? -1 : 0;
}

static void init_req(struct virt_llm_tensor_req *req)
{
	memset_(req, 0, sizeof(*req));
	req->abi = VIRT_LLM_TENSOR_ABI_VERSION;
	req->dtype = VIRT_LLM_DTYPE_F32;
}

static int submit_desc(int fd, u32 opcode, u32 backend, u32 command_id,
		       struct buf *input, struct buf *output,
		       struct buf *rsvd1, struct buf *rsvd2)
{
	struct virt_llm_user_desc desc;
	struct virt_llm_user_cpl cpl;

	memset_(&desc, 0, sizeof(desc));
	desc.opcode = opcode;
	desc.input_handle = input->ubuf.handle;
	desc.output_handle = output->ubuf.handle;
	desc.len = sizeof(struct virt_llm_tensor_req);
	desc.command_id = command_id;
	if (rsvd1)
		desc.rsvd1_handle = rsvd1->ubuf.handle;
	if (rsvd2)
		desc.rsvd2_handle = rsvd2->ubuf.handle;
	if (syscall3(29, fd, VIRT_LLM_IOCTL_SUBMIT_DESC, (long)&desc) < 0)
		return -1;
	if (wait_cq(fd, &cpl) < 0)
		return -1;
	if (cpl.command_id != command_id || cpl.backend != backend ||
	    cpl.status != VIRT_LLM_DESC_COMPLETE) {
		puts_("qwen op failed opcode=");
		print_hex(opcode);
		puts_(" status=");
		print_hex(cpl.status);
		puts_(" result=");
		print_hex(cpl.result);
		puts_("\n");
		return -1;
	}
	return 0;
}

static u32 layer_tensor(u32 layer, u32 slot)
{
	return T_LAYER_BASE + layer * T_LAYER_STRIDE + slot;
}

static int qwen_progress_failed(const char *stage, u32 layer)
{
	puts_("qwen progress failed stage=");
	puts_(stage);
	puts_(" layer=");
	print_dec(layer);
	puts_("\n");
	return -1;
}

static int model_load(int fd, struct buf *a, struct buf *b)
{
	struct virt_llm_user_desc desc;
	struct virt_llm_user_cpl cpl;

	memset_(&desc, 0, sizeof(desc));
	desc.opcode = VIRT_LLM_OP_MODEL_LOAD;
	desc.input_handle = a->ubuf.handle;
	desc.output_handle = b->ubuf.handle;
	desc.command_id = 1;
	if (syscall3(29, fd, VIRT_LLM_IOCTL_SUBMIT_DESC, (long)&desc) < 0)
		return -1;
	if (wait_cq(fd, &cpl) < 0)
		return -1;
	if (cpl.status != VIRT_LLM_DESC_COMPLETE || cpl.result != 290) {
		puts_("qwen model load failed status=");
		print_hex(cpl.status);
		puts_(" tensors=");
		print_dec(cpl.result);
		puts_("\n");
		return -1;
	}
	return 0;
}

static int embed(int fd, struct buf *tokens, struct buf *state,
		 u32 *token_ids, u32 seq)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)tokens->ptr;
	u32 *ids = (u32 *)((unsigned char *)tokens->ptr + DATA_OFF);

	memset_(tokens->ptr, 0, BUF_SIZE);
	init_req(req);
	req->rank = 2;
	req->tensor_id = T_EMBED;
	req->dims[0] = seq;
	req->dims[1] = QWEN_HIDDEN;
	req->dims[2] = QWEN_VOCAB;
	req->input_offset = DATA_OFF;
	req->output_offset = DATA_OFF;
	for (u32 i = 0; i < seq; i++)
		ids[i] = token_ids[i];
	return submit_desc(fd, VIRT_LLM_OP_EMBED_LOOKUP_F32, VIRT_LLM_BACKEND_VECTOR,
			   10, tokens, state, 0, 0);
}

static int rmsnorm(int fd, u32 command, struct buf *in, struct buf *out,
		   u32 tensor_id, u32 seq)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)in->ptr;

	init_req(req);
	req->rank = 2;
	req->tensor_id = tensor_id;
	req->dims[0] = seq;
	req->dims[1] = QWEN_HIDDEN;
	req->input_offset = DATA_OFF;
	req->output_offset = DATA_OFF;
	return submit_desc(fd, VIRT_LLM_OP_RMSNORM_F32, VIRT_LLM_BACKEND_VECTOR,
			   command, in, out, 0, 0);
}

static int gemm(int fd, u32 command, struct buf *in, struct buf *out,
		u32 tensor_id, u32 m, u32 n, u32 k, u32 opcode)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)in->ptr;

	init_req(req);
	req->rank = 2;
	req->tensor_id = tensor_id;
	req->dims[0] = m;
	req->dims[1] = n;
	req->dims[2] = k;
	req->input_offset = DATA_OFF;
	req->output_offset = DATA_OFF;
	return submit_desc(fd, opcode, VIRT_LLM_BACKEND_TENSOR, command, in, out, 0, 0);
}

static int gemm_bias(int fd, u32 command, struct buf *in, struct buf *out,
		     u32 tensor_id, u32 bias_id, u32 m, u32 n, u32 k)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)in->ptr;

	init_req(req);
	req->rank = 2;
	req->tensor_id = tensor_id;
	req->aux_tensor_id = bias_id;
	req->dims[0] = m;
	req->dims[1] = n;
	req->dims[2] = k;
	req->input_offset = DATA_OFF;
	req->output_offset = DATA_OFF;
	return submit_desc(fd, VIRT_LLM_OP_GEMM_F32, VIRT_LLM_BACKEND_TENSOR,
			   command, in, out, 0, 0);
}

static int lm_head(int fd, struct buf *in, struct buf *out, u32 seq)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)in->ptr;

	init_req(req);
	req->rank = 2;
	req->tensor_id = T_EMBED;
	req->dims[0] = 1;
	req->dims[1] = QWEN_VOCAB;
	req->dims[2] = QWEN_HIDDEN;
	req->input_offset = DATA_OFF + (seq - 1) * QWEN_HIDDEN * sizeof(u32);
	req->output_offset = DATA_OFF;
	return submit_desc(fd, VIRT_LLM_OP_LM_HEAD_F32, VIRT_LLM_BACKEND_TENSOR,
			   81, in, out, 0, 0);
}

static int rope(int fd, u32 command, struct buf *x, u32 seq, u32 heads)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)x->ptr;

	init_req(req);
	req->rank = 3;
	req->dims[0] = seq;
	req->dims[1] = heads;
	req->dims[2] = QWEN_HEAD_DIM;
	req->input_offset = DATA_OFF;
	req->output_offset = DATA_OFF;
	return submit_desc(fd, VIRT_LLM_OP_ROPE_F32, VIRT_LLM_BACKEND_VECTOR,
			   command, x, x, 0, 0);
}

static int gqa(int fd, struct buf *q, struct buf *k, struct buf *v,
	       struct buf *out, u32 seq)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)q->ptr;

	init_req(req);
	req->rank = 4;
	req->flags = VIRT_LLM_TENSOR_F_CAUSAL;
	req->dims[0] = seq;
	req->dims[1] = QWEN_HEADS;
	req->dims[2] = QWEN_KV_HEADS;
	req->dims[3] = QWEN_HEAD_DIM;
	req->input_offset = DATA_OFF;
	req->weight_offset = DATA_OFF;
	req->aux_offset = DATA_OFF;
	req->output_offset = DATA_OFF;
	return submit_desc(fd, VIRT_LLM_OP_QWEN_GQA_ATTENTION_F32,
			   VIRT_LLM_BACKEND_TENSOR, 40, q, out, k, v);
}

static int add(int fd, u32 command, struct buf *state, struct buf *delta, u32 seq)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)state->ptr;

	init_req(req);
	req->rank = 2;
	req->dims[0] = seq;
	req->dims[1] = QWEN_HIDDEN;
	req->input_offset = DATA_OFF;
	req->input2_offset = DATA_OFF;
	req->output_offset = DATA_OFF;
	return submit_desc(fd, VIRT_LLM_OP_ADD_F32, VIRT_LLM_BACKEND_VECTOR,
			   command, state, state, delta, 0);
}

static int swiglu(int fd, struct buf *gate, struct buf *up, u32 seq)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)gate->ptr;

	init_req(req);
	req->rank = 2;
	req->dims[0] = seq;
	req->dims[1] = QWEN_INTERMEDIATE;
	req->input_offset = DATA_OFF;
	req->input2_offset = DATA_OFF;
	req->output_offset = DATA_OFF;
	return submit_desc(fd, VIRT_LLM_OP_SWIGLU_F32, VIRT_LLM_BACKEND_VECTOR,
			   70, gate, gate, up, 0);
}

static int argmax(int fd, struct buf *logits, struct buf *token, u32 seq)
{
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)logits->ptr;

	(void)seq;
	init_req(req);
	req->rank = 1;
	req->dims[0] = QWEN_VOCAB;
	req->input_offset = DATA_OFF;
	req->output_offset = DATA_OFF;
	return submit_desc(fd, VIRT_LLM_OP_ARGMAX_F32, VIRT_LLM_BACKEND_VECTOR,
			   90, logits, token, 0, 0);
}

static int run_qwen_forward(int fd, struct buf *b, u32 *token_ids, u32 seq,
			    u32 *next_token)
{
	if (embed(fd, &b[1], &b[0], token_ids, seq) < 0)
		return qwen_progress_failed("embed", 0);
	puts_("qwen embed ok\n");
	for (u32 layer = 0; layer < QWEN_LAYER_COUNT; layer++) {
		if (rmsnorm(fd, 20, &b[0], &b[1],
			    layer_tensor(layer, T_INPUT_NORM), seq) < 0)
			return qwen_progress_failed("input_norm", layer);
		if (gemm_bias(fd, 30, &b[1], &b[2], layer_tensor(layer, T_Q_PROJ),
			      layer_tensor(layer, T_Q_BIAS), seq, QWEN_HIDDEN,
			      QWEN_HIDDEN) < 0)
			return qwen_progress_failed("q_proj", layer);
		if (gemm_bias(fd, 31, &b[1], &b[3], layer_tensor(layer, T_K_PROJ),
			      layer_tensor(layer, T_K_BIAS), seq,
			      QWEN_KV_HEADS * QWEN_HEAD_DIM, QWEN_HIDDEN) < 0)
			return qwen_progress_failed("k_proj", layer);
		if (gemm_bias(fd, 32, &b[1], &b[4], layer_tensor(layer, T_V_PROJ),
			      layer_tensor(layer, T_V_BIAS), seq,
			      QWEN_KV_HEADS * QWEN_HEAD_DIM, QWEN_HIDDEN) < 0)
			return qwen_progress_failed("v_proj", layer);
		if (rope(fd, 33, &b[2], seq, QWEN_HEADS) < 0 ||
		    rope(fd, 34, &b[3], seq, QWEN_KV_HEADS) < 0)
			return qwen_progress_failed("rope", layer);
		if (gqa(fd, &b[2], &b[3], &b[4], &b[5], seq) < 0)
			return qwen_progress_failed("gqa", layer);
		if (gemm(fd, 50, &b[5], &b[6], layer_tensor(layer, T_O_PROJ),
			 seq, QWEN_HIDDEN, QWEN_HIDDEN, VIRT_LLM_OP_GEMM_F32) < 0)
			return qwen_progress_failed("o_proj", layer);
		if (add(fd, 51, &b[0], &b[6], seq) < 0)
			return qwen_progress_failed("attn_residual", layer);
		if (rmsnorm(fd, 60, &b[0], &b[1],
			    layer_tensor(layer, T_POST_NORM), seq) < 0)
			return qwen_progress_failed("post_norm", layer);
		if (gemm(fd, 61, &b[1], &b[2], layer_tensor(layer, T_GATE_PROJ),
			 seq, QWEN_INTERMEDIATE, QWEN_HIDDEN, VIRT_LLM_OP_GEMM_F32) < 0)
			return qwen_progress_failed("gate_proj", layer);
		if (gemm(fd, 62, &b[1], &b[3], layer_tensor(layer, T_UP_PROJ),
			 seq, QWEN_INTERMEDIATE, QWEN_HIDDEN, VIRT_LLM_OP_GEMM_F32) < 0)
			return qwen_progress_failed("up_proj", layer);
		if (swiglu(fd, &b[2], &b[3], seq) < 0)
			return qwen_progress_failed("swiglu", layer);
		if (gemm(fd, 71, &b[2], &b[6], layer_tensor(layer, T_DOWN_PROJ),
			 seq, QWEN_HIDDEN, QWEN_INTERMEDIATE, VIRT_LLM_OP_GEMM_F32) < 0)
			return qwen_progress_failed("down_proj", layer);
		if (add(fd, 72, &b[0], &b[6], seq) < 0)
			return qwen_progress_failed("mlp_residual", layer);
	}
	puts_("qwen full layers ok\n");
	if (rmsnorm(fd, 80, &b[0], &b[1], T_FINAL_NORM, seq) < 0)
		return qwen_progress_failed("final_norm", QWEN_LAYER_COUNT);
	if (lm_head(fd, &b[1], &b[5], seq) < 0)
		return qwen_progress_failed("lm_head", QWEN_LAYER_COUNT);
	if (argmax(fd, &b[5], &b[7], seq) < 0)
		return qwen_progress_failed("argmax", QWEN_LAYER_COUNT);
	*next_token = *(u32 *)((unsigned char *)b[7].ptr + DATA_OFF);
	return 0;
}

static int run_qwen_decode(int fd, struct buf *b)
{
	u32 token_ids[QWEN_MAX_CONTEXT];
	u32 output_tokens[QWEN_DECODE_STEPS];

	if (model_load(fd, &b[0], &b[1]) < 0)
		return -1;
	puts_("qwen model load ok\n");
	puts_("qwen fixture prompt=");
	puts_(QWEN_FIXTURE_PROMPT);
	puts_("\n");
	puts_("qwen fixture input_tokens=");
	print_token_list(qwen_prompt_tokens, QWEN_PROMPT_TOKEN_COUNT);
	puts_("\n");
	for (u32 i = 0; i < QWEN_PROMPT_TOKEN_COUNT; i++)
		token_ids[i] = qwen_prompt_tokens[i];
	for (u32 step = 0; step < QWEN_DECODE_STEPS; step++) {
		u32 next = 0;
		u32 seq = QWEN_PROMPT_TOKEN_COUNT + step;

		if (run_qwen_forward(fd, b, token_ids, seq, &next) < 0)
			return -1;
		token_ids[seq] = next;
		output_tokens[step] = next;
		puts_("qwen decode step token=");
		print_dec(next);
		puts_("\n");
	}
	if (output_tokens[0] != QWEN_EXPECTED_FIRST_TOKEN) {
		puts_("QWEN_INFER_FAILED expected_first=");
		print_dec(QWEN_EXPECTED_FIRST_TOKEN);
		puts_(" current_first=");
		print_dec(output_tokens[0]);
		puts_(" input_tokens=");
		print_token_list(qwen_prompt_tokens, QWEN_PROMPT_TOKEN_COUNT);
		puts_(" output_tokens=");
		print_token_list(output_tokens, QWEN_DECODE_STEPS);
		puts_("\n");
		return -1;
	}
	puts_("QWEN_SINGLE_TOKEN_OK token=");
	print_dec(output_tokens[0]);
	puts_("\n");
	puts_("QWEN_DECODE_OK tokens=");
	print_token_list(output_tokens, QWEN_DECODE_STEPS);
	puts_("\n");
	puts_("QWEN_INFER_OK input_tokens=");
	print_token_list(qwen_prompt_tokens, QWEN_PROMPT_TOKEN_COUNT);
	puts_(" output_tokens=");
	print_token_list(output_tokens, QWEN_DECODE_STEPS);
	puts_("\n");
	return 0;
}

void _start(void)
{
	struct buf bufs[8];
	int fd = open_dev();
	int ok = 0;

	if (fd < 0) {
		puts_("qwen open failed\n");
	} else {
		ok = 1;
		for (u32 i = 0; i < 8; i++) {
			if (alloc_buffer(fd, &bufs[i]) < 0) {
				ok = 0;
				break;
			}
		}
		if (ok && run_qwen_decode(fd, bufs) == 0)
			syscall4(142, 0xfee1dead, 672274793, 0x1234567, 0);
	}

	puts_("QWEN_INFER_FAILED\n");
	puts_("QWEN_SINGLE_TOKEN_FAILED\n");
	syscall1(93, 1);
	for (;;)
		syscall0(93);
}
