// SPDX-License-Identifier: GPL-2.0-only
/*
 * Freestanding riscv32 /init smoke test for /dev/virt_llm0.
 */

typedef unsigned int u32;
typedef unsigned long ulong;
typedef unsigned long long u64;

#define AT_FDCWD        -100
#define O_RDWR          02
#define S_IFCHR         0020000
#define PROT_READ       0x1
#define PROT_WRITE      0x2
#define MAP_SHARED      0x01
#define PAGE_SIZE       4096

#define VIRT_LLM_IOCTL_MAGIC 'L'
#define VIRT_LLM_OP_GEMM_U32       0x0200
#define VIRT_LLM_OP_ATTENTION_Q16  0x0202
#define VIRT_LLM_OP_MODEL_QUERY    0x0301
#define VIRT_LLM_OP_GEMM_F32       0x0313
#define VIRT_LLM_DESC_COMPLETE     1
#define VIRT_LLM_BACKEND_DMA       1
#define VIRT_LLM_BACKEND_TENSOR    3
#define VIRT_LLM_ATTENTION_CAUSAL  (1U << 0)
#define VIRT_LLM_TENSOR_ABI_VERSION 1
#define VIRT_LLM_DTYPE_F32         2
#define VIRT_LLM_REQ_DATA_OFFSET   128

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
#define _IOW(type, nr, size) _IOC(_IOC_WRITE, (type), (nr), sizeof(size))
#define _IOWR(type, nr, size) _IOC(_IOC_READ | _IOC_WRITE, (type), (nr), sizeof(size))

struct virt_llm_user_info {
	u32 magic;
	u32 version;
	u32 features;
	u32 abi;
	u32 q_max;
	u32 xfer_max;
	u32 cq_size;
	u32 reserved;
};

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

struct virt_llm_model_query {
	u32 abi;
	u32 model_loaded;
	u32 layers;
	u32 hidden_size;
	u32 attention_heads;
	u32 kv_heads;
	u32 head_dim;
	u32 intermediate_size;
	u32 vocab_size;
	u32 dtype;
	u64 rope_theta_bits;
	u64 rms_eps_bits;
};

#define VIRT_LLM_IOCTL_GET_INFO \
	_IOR(VIRT_LLM_IOCTL_MAGIC, 0x00, struct virt_llm_user_info)
#define VIRT_LLM_IOCTL_ALLOC_BUFFER \
	_IOWR(VIRT_LLM_IOCTL_MAGIC, 0x01, struct virt_llm_user_buffer)
#define VIRT_LLM_IOCTL_SUBMIT_DESC \
	_IOWR(VIRT_LLM_IOCTL_MAGIC, 0x03, struct virt_llm_user_desc)
#define VIRT_LLM_IOCTL_WAIT_CQ \
	_IOR(VIRT_LLM_IOCTL_MAGIC, 0x04, struct virt_llm_user_cpl)

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

static void memcpy_(void *dst, const void *src, unsigned int n)
{
	unsigned char *d = dst;
	const unsigned char *s = src;

	for (unsigned int i = 0; i < n; i++)
		d[i] = s[i];
}

void *memcpy(void *dst, const void *src, unsigned int n)
{
	memcpy_(dst, src, n);
	return dst;
}

void *memset(void *p, int v, unsigned int n)
{
	return memset_(p, v, n);
}

static void print_hex(u32 v)
{
	char out[] = "0x00000000";
	const char h[] = "0123456789abcdef";

	for (int i = 0; i < 8; i++)
		out[2 + i] = h[(v >> (28 - i * 4)) & 0xf];
	puts_(out);
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

static u32 devno(unsigned int major, unsigned int minor)
{
	return ((minor & 0xff) | (major << 8) | ((minor & ~0xff) << 12));
}

static int open_dev(void)
{
	long rc;

	syscall3(34, AT_FDCWD, (long)"/dev", 0755);
	rc = syscall4(33, AT_FDCWD, (long)"/dev/virt_llm0", S_IFCHR | 0600,
		      devno(10, 243));
	if (rc < 0)
		puts_("virt-llm-test mknod failed\n");
	return syscall4(56, AT_FDCWD, (long)"/dev/virt_llm0", O_RDWR, 0);
}

static int alloc_buffer(int fd, struct virt_llm_user_buffer *buf, u32 **ptr)
{
	ulong addr;

	memset_(buf, 0, sizeof(*buf));
	buf->size = PAGE_SIZE;
	if (syscall3(29, fd, VIRT_LLM_IOCTL_ALLOC_BUFFER, (long)buf) < 0) {
		puts_("virt-llm-test alloc ioctl failed\n");
		return -1;
	}
	addr = (ulong)syscall6(222, 0, buf->size, PROT_READ | PROT_WRITE, MAP_SHARED,
			       fd, buf->handle - 1);
	if (addr >= (ulong)-4095) {
		puts_("virt-llm-test mmap failed rc=");
		print_hex((u32)addr);
		puts_("\n");
		return -1;
	}
	*ptr = (u32 *)addr;
	return 0;
}

static int wait_cq(int fd, struct virt_llm_user_cpl *cpl)
{
	memset_(cpl, 0, sizeof(*cpl));
	return syscall3(29, fd, VIRT_LLM_IOCTL_WAIT_CQ, (long)cpl) < 0 ? -1 : 0;
}

static int run_info(int fd)
{
	struct virt_llm_user_info info;

	memset_(&info, 0, sizeof(info));
	if (syscall3(29, fd, VIRT_LLM_IOCTL_GET_INFO, (long)&info) < 0) {
		puts_("virt-llm-test info ioctl failed\n");
		return -1;
	}
	puts_("virt-llm-test info ok: magic=");
	print_hex(info.magic);
	puts_(" version=");
	print_dec(info.version);
	puts_("\n");
	return 0;
}

static int run_model_query(int fd, struct virt_llm_user_buffer *in_buf, u32 *in,
			   struct virt_llm_user_buffer *out_buf, u32 *out)
{
	struct virt_llm_user_desc desc;
	struct virt_llm_user_cpl cpl;
	struct virt_llm_model_query *query = (struct virt_llm_model_query *)out;

	memset_(in, 0, PAGE_SIZE);
	memset_(out, 0, PAGE_SIZE);
	memset_(&desc, 0, sizeof(desc));
	desc.opcode = VIRT_LLM_OP_MODEL_QUERY;
	desc.input_handle = in_buf->handle;
	desc.output_handle = out_buf->handle;
	desc.command_id = 2100;
	if (syscall3(29, fd, VIRT_LLM_IOCTL_SUBMIT_DESC, (long)&desc) < 0) {
		puts_("virt-llm-test model query submit failed\n");
		return -1;
	}
	if (wait_cq(fd, &cpl) < 0) {
		puts_("virt-llm-test model query wait failed\n");
		return -1;
	}
	if (cpl.command_id != 2100 || cpl.backend != VIRT_LLM_BACKEND_DMA ||
	    cpl.status != VIRT_LLM_DESC_COMPLETE || query->layers != 24 ||
	    query->hidden_size != 896 || query->attention_heads != 14 ||
	    query->kv_heads != 2 || query->head_dim != 64 ||
	    query->intermediate_size != 4864 || query->vocab_size != 151936) {
		puts_("virt-llm-test model query mismatch\n");
		return -1;
	}
	puts_("virt-llm-test model query ok: hidden=");
	print_dec(query->hidden_size);
	puts_(" layers=");
	print_dec(query->layers);
	puts_("\n");
	return 0;
}

static int run_gemm_f32(int fd, struct virt_llm_user_buffer *a_buf, u32 *a_raw,
			struct virt_llm_user_buffer *b_buf, u32 *b_raw,
			struct virt_llm_user_buffer *out_buf, u32 *out_raw)
{
	struct virt_llm_user_desc desc;
	struct virt_llm_user_cpl cpl;
	struct virt_llm_tensor_req *req = (struct virt_llm_tensor_req *)a_raw;
	u32 *a = (u32 *)((unsigned char *)a_raw + VIRT_LLM_REQ_DATA_OFFSET);
	u32 *b = b_raw;
	u32 expected[] = { 0x41980000, 0x41b00000, 0x422c0000, 0x42480000 };

	memset_(a_raw, 0, PAGE_SIZE);
	memset_(b_raw, 0, PAGE_SIZE);
	memset_(out_raw, 0, PAGE_SIZE);
	a[0] = 0x3f800000; a[1] = 0x40000000;
	a[2] = 0x40400000; a[3] = 0x40800000;
	b[0] = 0x40a00000; b[1] = 0x40c00000;
	b[2] = 0x40e00000; b[3] = 0x41000000;
	req->abi = VIRT_LLM_TENSOR_ABI_VERSION;
	req->dtype = VIRT_LLM_DTYPE_F32;
	req->rank = 2;
	req->dims[0] = 2;
	req->dims[1] = 2;
	req->dims[2] = 2;
	req->input_offset = VIRT_LLM_REQ_DATA_OFFSET;
	req->weight_offset = 0;
	req->output_offset = 0;

	memset_(&desc, 0, sizeof(desc));
	desc.opcode = VIRT_LLM_OP_GEMM_F32;
	desc.input_handle = a_buf->handle;
	desc.output_handle = out_buf->handle;
	desc.len = sizeof(*req);
	desc.command_id = 2101;
	desc.rsvd1_handle = b_buf->handle;
	if (syscall3(29, fd, VIRT_LLM_IOCTL_SUBMIT_DESC, (long)&desc) < 0) {
		puts_("virt-llm-test gemm f32 submit failed\n");
		return -1;
	}
	if (wait_cq(fd, &cpl) < 0) {
		puts_("virt-llm-test gemm f32 wait failed\n");
		return -1;
	}
	if (cpl.command_id != 2101 || cpl.backend != VIRT_LLM_BACKEND_TENSOR ||
	    cpl.status != VIRT_LLM_DESC_COMPLETE) {
		puts_("virt-llm-test gemm f32 cpl mismatch\n");
		return -1;
	}
	for (unsigned int i = 0; i < 4; i++) {
		if (out_raw[i] != expected[i])
			return -1;
	}
	puts_("virt-llm-test gemm f32 ok: checksum=");
	print_hex(cpl.result);
	puts_("\n");
	return 0;
}

static int run_gemm(int fd, struct virt_llm_user_buffer *a_buf, u32 *a,
		    struct virt_llm_user_buffer *b_buf, u32 *b,
		    struct virt_llm_user_buffer *out_buf, u32 *out)
{
	struct virt_llm_user_desc desc;
	struct virt_llm_user_cpl cpl;
	u32 expected[] = { 19, 22, 43, 50 };
	u64 dims = 2 | (2ULL << 16) | (2ULL << 32);

	a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
	b[0] = 5; b[1] = 6; b[2] = 7; b[3] = 8;
	memset_(out, 0, PAGE_SIZE);
	memset_(&desc, 0, sizeof(desc));
	desc.opcode = VIRT_LLM_OP_GEMM_U32;
	desc.input_handle = a_buf->handle;
	desc.output_handle = out_buf->handle;
	desc.command_id = 2001;
	desc.rsvd1_handle = b_buf->handle;
	desc.rsvd2_addr = dims;
	if (syscall3(29, fd, VIRT_LLM_IOCTL_SUBMIT_DESC, (long)&desc) < 0) {
		puts_("virt-llm-test gemm submit failed\n");
		return -1;
	}
	if (wait_cq(fd, &cpl) < 0) {
		puts_("virt-llm-test gemm wait failed\n");
		return -1;
	}
	if (cpl.command_id != 2001 || cpl.backend != VIRT_LLM_BACKEND_TENSOR ||
	    cpl.status != VIRT_LLM_DESC_COMPLETE || cpl.result != 134) {
		puts_("virt-llm-test gemm cpl mismatch result=");
		print_hex(cpl.result);
		puts_("\n");
		return -1;
	}
	for (unsigned int i = 0; i < 4; i++) {
		if (out[i] != expected[i])
			return -1;
	}
	puts_("virt-llm-test gemm ok: checksum=");
	print_hex(cpl.result);
	puts_("\n");
	return 0;
}

static int run_attention(int fd, struct virt_llm_user_buffer *q_buf, u32 *q,
			 struct virt_llm_user_buffer *k_buf, u32 *k,
			 struct virt_llm_user_buffer *v_buf, u32 *v,
			 struct virt_llm_user_buffer *out_buf, u32 *out)
{
	struct virt_llm_user_desc desc;
	struct virt_llm_user_cpl cpl;
	u32 q_values[] = { 131072, 0, 0, 131072, 131072, 131072 };
	u32 k_values[] = { 131072, 0, 0, 131072, 131072, 131072 };
	u32 v_values[] = { 65536, 0, 0, 65536, 65536, 65536 };
	u32 expected[] = { 65536, 0, 1179, 64356, 64376, 64376 };

	memset_(q, 0, PAGE_SIZE);
	memset_(k, 0, PAGE_SIZE);
	memset_(v, 0, PAGE_SIZE);
	memset_(out, 0, PAGE_SIZE);
	memcpy_(q, q_values, sizeof(q_values));
	memcpy_(k, k_values, sizeof(k_values));
	memcpy_(v, v_values, sizeof(v_values));
	memset_(&desc, 0, sizeof(desc));
	desc.opcode = VIRT_LLM_OP_ATTENTION_Q16;
	desc.input_handle = q_buf->handle;
	desc.output_handle = out_buf->handle;
	desc.len = 3 | (2U << 16);
	desc.command_id = 2002;
	desc.rsvd1_handle = k_buf->handle;
	desc.rsvd2_handle = v_buf->handle;
	desc.rsvd3 = VIRT_LLM_ATTENTION_CAUSAL;
	if (syscall3(29, fd, VIRT_LLM_IOCTL_SUBMIT_DESC, (long)&desc) < 0) {
		puts_("virt-llm-test attention submit failed\n");
		return -1;
	}
	if (wait_cq(fd, &cpl) < 0) {
		puts_("virt-llm-test attention wait failed\n");
		return -1;
	}
	if (cpl.command_id != 2002 || cpl.backend != VIRT_LLM_BACKEND_TENSOR ||
	    cpl.status != VIRT_LLM_DESC_COMPLETE || cpl.result != 0x3f6ef) {
		puts_("virt-llm-test attention cpl mismatch result=");
		print_hex(cpl.result);
		puts_("\n");
		return -1;
	}
	for (unsigned int i = 0; i < 6; i++) {
		if (out[i] != expected[i])
			return -1;
	}
	puts_("virt-llm-test attention ok: checksum=");
	print_hex(cpl.result);
	puts_("\n");
	return 0;
}

void _start(void)
{
	struct virt_llm_user_buffer a_buf, b_buf, c_buf, out_buf;
	u32 *a, *b, *c, *out;
	int fd = open_dev();
	int ok = 0;

	if (fd < 0)
		puts_("virt-llm-test open failed\n");
	else if (alloc_buffer(fd, &a_buf, &a) == 0 &&
		 alloc_buffer(fd, &b_buf, &b) == 0 &&
		 alloc_buffer(fd, &c_buf, &c) == 0 &&
		 alloc_buffer(fd, &out_buf, &out) == 0 &&
		 run_info(fd) == 0 &&
		 run_model_query(fd, &a_buf, a, &out_buf, out) == 0 &&
		 run_gemm(fd, &a_buf, a, &b_buf, b, &out_buf, out) == 0 &&
		 run_attention(fd, &a_buf, a, &b_buf, b, &c_buf, c, &out_buf, out) == 0 &&
		 run_gemm_f32(fd, &a_buf, a, &b_buf, b, &out_buf, out) == 0)
		ok = 1;

	if (ok) {
		puts_("INITRAMFS_OK: Linux 6.12 booted on QEMU riscv32\n");
		syscall4(142, 0xfee1dead, 672274793, 0x1234567, 0);
	} else {
		puts_("virt-llm-test failed\n");
		syscall1(93, 1);
	}
	for (;;)
		syscall0(93);
}
