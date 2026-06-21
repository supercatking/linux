// SPDX-License-Identifier: GPL-2.0-only
/*
 * Freestanding riscv32 interactive console for /dev/virt_llm0.
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
#define PAGE_SIZE       4096

#define VIRT_LLM_IOCTL_MAGIC 'L'
#define VIRT_LLM_OP_GEMM_U32       0x0200
#define VIRT_LLM_OP_ATTENTION_Q16  0x0202
#define VIRT_LLM_DESC_COMPLETE     1
#define VIRT_LLM_BACKEND_TENSOR    3
#define VIRT_LLM_ATTENTION_CAUSAL  (1U << 0)

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

struct virt_llm_ctx {
	int fd;
	struct virt_llm_user_buffer a_buf;
	struct virt_llm_user_buffer b_buf;
	struct virt_llm_user_buffer c_buf;
	struct virt_llm_user_buffer out_buf;
	u32 *a;
	u32 *b;
	u32 *c;
	u32 *out;
};

#define VIRT_LLM_IOCTL_GET_INFO \
	_IOR(VIRT_LLM_IOCTL_MAGIC, 0x00, struct virt_llm_user_info)
#define VIRT_LLM_IOCTL_ALLOC_BUFFER \
	_IOWR(VIRT_LLM_IOCTL_MAGIC, 0x01, struct virt_llm_user_buffer)
#define VIRT_LLM_IOCTL_SUBMIT_DESC \
	_IOWR(VIRT_LLM_IOCTL_MAGIC, 0x03, struct virt_llm_user_desc)
#define VIRT_LLM_IOCTL_WAIT_CQ \
	_IOR(VIRT_LLM_IOCTL_MAGIC, 0x04, struct virt_llm_user_cpl)

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

static int streq(const char *a, const char *b)
{
	unsigned int i = 0;

	while (a[i] && b[i]) {
		if (a[i] != b[i])
			return 0;
		i++;
	}
	return a[i] == b[i];
}

static void strip_line(char *buf)
{
	for (unsigned int i = 0; buf[i]; i++) {
		if (buf[i] == '\r' || buf[i] == '\n') {
			buf[i] = 0;
			return;
		}
	}
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

static int alloc_buffer(int fd, struct virt_llm_user_buffer *buf, u32 **ptr)
{
	ulong addr;

	memset_(buf, 0, sizeof(*buf));
	buf->size = PAGE_SIZE;
	if (syscall3(29, fd, VIRT_LLM_IOCTL_ALLOC_BUFFER, (long)buf) < 0) {
		puts_("alloc buffer failed\n");
		return -1;
	}
	addr = (ulong)syscall6(222, 0, buf->size, PROT_READ | PROT_WRITE, MAP_SHARED,
			       fd, buf->handle - 1);
	if (addr >= (ulong)-4095) {
		puts_("mmap failed rc=");
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

static int ctx_init(struct virt_llm_ctx *ctx)
{
	memset_(ctx, 0, sizeof(*ctx));
	ctx->fd = open_dev();
	if (ctx->fd < 0) {
		puts_("open /dev/virt_llm0 failed\n");
		return -1;
	}
	if (alloc_buffer(ctx->fd, &ctx->a_buf, &ctx->a) < 0 ||
	    alloc_buffer(ctx->fd, &ctx->b_buf, &ctx->b) < 0 ||
	    alloc_buffer(ctx->fd, &ctx->c_buf, &ctx->c) < 0 ||
	    alloc_buffer(ctx->fd, &ctx->out_buf, &ctx->out) < 0)
		return -1;
	return 0;
}

static int run_info(struct virt_llm_ctx *ctx)
{
	struct virt_llm_user_info info;

	memset_(&info, 0, sizeof(info));
	if (syscall3(29, ctx->fd, VIRT_LLM_IOCTL_GET_INFO, (long)&info) < 0) {
		puts_("info ioctl failed\n");
		return -1;
	}
	puts_("info ok: magic=");
	print_hex(info.magic);
	puts_(" version=");
	print_dec(info.version);
	puts_(" abi=");
	print_dec(info.abi);
	puts_(" features=");
	print_hex(info.features);
	puts_(" q_max=");
	print_dec(info.q_max);
	puts_(" xfer_max=");
	print_dec(info.xfer_max);
	puts_("\n");
	return 0;
}

static int run_gemm(struct virt_llm_ctx *ctx)
{
	struct virt_llm_user_desc desc;
	struct virt_llm_user_cpl cpl;
	u32 expected[] = { 19, 22, 43, 50 };
	u64 dims = 2 | (2ULL << 16) | (2ULL << 32);

	ctx->a[0] = 1; ctx->a[1] = 2; ctx->a[2] = 3; ctx->a[3] = 4;
	ctx->b[0] = 5; ctx->b[1] = 6; ctx->b[2] = 7; ctx->b[3] = 8;
	memset_(ctx->out, 0, PAGE_SIZE);
	memset_(&desc, 0, sizeof(desc));
	desc.opcode = VIRT_LLM_OP_GEMM_U32;
	desc.input_handle = ctx->a_buf.handle;
	desc.output_handle = ctx->out_buf.handle;
	desc.command_id = 3001;
	desc.rsvd1_handle = ctx->b_buf.handle;
	desc.rsvd2_addr = dims;
	if (syscall3(29, ctx->fd, VIRT_LLM_IOCTL_SUBMIT_DESC, (long)&desc) < 0) {
		puts_("gemm submit failed\n");
		return -1;
	}
	if (wait_cq(ctx->fd, &cpl) < 0) {
		puts_("gemm wait failed\n");
		return -1;
	}
	if (cpl.command_id != 3001 || cpl.backend != VIRT_LLM_BACKEND_TENSOR ||
	    cpl.status != VIRT_LLM_DESC_COMPLETE || cpl.result != 134) {
		puts_("gemm cpl mismatch result=");
		print_hex(cpl.result);
		puts_(" status=");
		print_hex(cpl.status);
		puts_("\n");
		return -1;
	}
	for (unsigned int i = 0; i < 4; i++) {
		if (ctx->out[i] != expected[i]) {
			puts_("gemm output mismatch\n");
			return -1;
		}
	}
	puts_("gemm ok: checksum=");
	print_hex(cpl.result);
	puts_(" output=[19,22,43,50]\n");
	return 0;
}

static int run_attention(struct virt_llm_ctx *ctx)
{
	struct virt_llm_user_desc desc;
	struct virt_llm_user_cpl cpl;
	u32 q_values[] = { 131072, 0, 0, 131072, 131072, 131072 };
	u32 k_values[] = { 131072, 0, 0, 131072, 131072, 131072 };
	u32 v_values[] = { 65536, 0, 0, 65536, 65536, 65536 };
	u32 expected[] = { 65536, 0, 1179, 64356, 64376, 64376 };

	memset_(ctx->a, 0, PAGE_SIZE);
	memset_(ctx->b, 0, PAGE_SIZE);
	memset_(ctx->c, 0, PAGE_SIZE);
	memset_(ctx->out, 0, PAGE_SIZE);
	memcpy_(ctx->a, q_values, sizeof(q_values));
	memcpy_(ctx->b, k_values, sizeof(k_values));
	memcpy_(ctx->c, v_values, sizeof(v_values));
	memset_(&desc, 0, sizeof(desc));
	desc.opcode = VIRT_LLM_OP_ATTENTION_Q16;
	desc.input_handle = ctx->a_buf.handle;
	desc.output_handle = ctx->out_buf.handle;
	desc.len = 3 | (2U << 16);
	desc.command_id = 3002;
	desc.rsvd1_handle = ctx->b_buf.handle;
	desc.rsvd2_handle = ctx->c_buf.handle;
	desc.rsvd3 = VIRT_LLM_ATTENTION_CAUSAL;
	if (syscall3(29, ctx->fd, VIRT_LLM_IOCTL_SUBMIT_DESC, (long)&desc) < 0) {
		puts_("attention submit failed\n");
		return -1;
	}
	if (wait_cq(ctx->fd, &cpl) < 0) {
		puts_("attention wait failed\n");
		return -1;
	}
	if (cpl.command_id != 3002 || cpl.backend != VIRT_LLM_BACKEND_TENSOR ||
	    cpl.status != VIRT_LLM_DESC_COMPLETE || cpl.result != 0x3f6ef) {
		puts_("attention cpl mismatch result=");
		print_hex(cpl.result);
		puts_(" status=");
		print_hex(cpl.status);
		puts_("\n");
		return -1;
	}
	for (unsigned int i = 0; i < 6; i++) {
		if (ctx->out[i] != expected[i]) {
			puts_("attention output mismatch\n");
			return -1;
		}
	}
	puts_("attention ok: checksum=");
	print_hex(cpl.result);
	puts_(" seq=3 head_dim=2 causal=1\n");
	return 0;
}

static int read_line(char *buf, unsigned int size)
{
	unsigned int pos = 0;
	char ch;

	while (pos + 1 < size) {
		long rc = syscall3(63, 0, (long)&ch, 1);

		if (rc <= 0)
			return -1;
		if (ch == '\r')
			continue;
		if (ch == '\n') {
			puts_("\n");
			break;
		}
		if (ch == 0x7f || ch == '\b') {
			if (pos) {
				pos--;
				puts_("\b \b");
			}
			continue;
		}
		buf[pos++] = ch;
		syscall3(64, 1, (long)&ch, 1);
	}
	buf[pos] = 0;
	strip_line(buf);
	return pos;
}

static void print_help(void)
{
	puts_("commands:\n");
	puts_("  help       show this text\n");
	puts_("  info       read virt-llm device info\n");
	puts_("  gemm       submit toy GEMM_U32 command\n");
	puts_("  attention  submit toy ATTENTION_Q16 command\n");
	puts_("  reboot     reboot guest\n");
}

void _start(void)
{
	struct virt_llm_ctx ctx;
	char line[64];

	puts_("virt-llm interactive console\n");
	puts_("type 'help' for commands\n");
	if (ctx_init(&ctx) < 0) {
		puts_("console init failed\n");
		syscall1(93, 1);
	}

	for (;;) {
		puts_("virt-llm> ");
		if (read_line(line, sizeof(line)) < 0)
			continue;
		if (!line[0])
			continue;
		if (streq(line, "help")) {
			print_help();
		} else if (streq(line, "info")) {
			run_info(&ctx);
		} else if (streq(line, "gemm")) {
			run_gemm(&ctx);
		} else if (streq(line, "attention")) {
			run_attention(&ctx);
		} else if (streq(line, "reboot")) {
			puts_("rebooting\n");
			syscall4(142, 0xfee1dead, 672274793, 0x1234567, 0);
		} else {
			puts_("unknown command: ");
			puts_(line);
			puts_("\n");
		}
	}
}
