/* ============================================================================
 * File   : clai_kv260_uio.h
 * Target : Kria KV260 Zynq UltraScale+ PS (Ubuntu 24.04 / PetaLinux `/dev/uioX`)
 * License: Apache-2.0 / MIT
 * Summary: Zero-copy POSIX C11/C++20 Linux UIO userspace driver for `clai_kv260_axi_top`
 *          handling `irq_escalate_to_ps` interrupts, 512-bit EngramBitBus readback,
 *          Engram BRAM updates, and 4.0 ns glitch-free Shadow-Bank LUT6 hot-swaps.
 * ============================================================================ */

#ifndef CLAI_KV260_UIO_H
#define CLAI_KV260_UIO_H

#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CLAI_KV260_MAP_BYTES       0x1000U
#define CLAI_KV260_MAGIC           0x434C4149U /* "CLAI" */

/* AXI4-Lite Byte Register Offsets */
#define CLAI_CSR_CTRL              0x000U
#define CLAI_CSR_STATUS            0x004U
#define CLAI_CSR_MARGIN_THRESH     0x008U
#define CLAI_CSR_RESULT            0x00CU
#define CLAI_CSR_TOTAL_EPISODES    0x010U
#define CLAI_CSR_SATIATED_CNT      0x014U
#define CLAI_CSR_ESCALATED_CNT     0x018U
#define CLAI_CSR_CLASS_VOTES_BASE  0x020U /* 0x020..0x03C (8 classes) */
#define CLAI_CSR_BRAM_CTRL         0x040U
#define CLAI_CSR_BRAM_WDATA_LO     0x044U
#define CLAI_CSR_BRAM_WDATA_HI     0x048U
#define CLAI_CSR_HS_CTRL           0x050U
#define CLAI_CSR_HS_TABLE_LO       0x054U
#define CLAI_CSR_HS_TABLE_HI       0x058U
#define CLAI_CSR_HS_WIRES_LO       0x05CU
#define CLAI_CSR_HS_WIRES_HI       0x060U
#define CLAI_CSR_HS_GENERATION     0x064U
#define CLAI_CSR_MAGIC             0x080U
#define CLAI_CSR_BUS512_BASE       0x100U /* 0x100..0x13C (16 x uint32_t) */

/* CSR_CTRL Bit Masks */
#define CLAI_CTRL_SOFT_TRIGGER     (1U << 0)
#define CLAI_CTRL_USE_MMIO_BUS     (1U << 1)
#define CLAI_CTRL_IRQ_ENABLE       (1U << 2)
#define CLAI_CTRL_CLEAR_COUNTERS   (1U << 3)
#define CLAI_CTRL_OVERLAY_ENABLE   (1U << 4)
#define CLAI_CTRL_ATOMIC_BANK_SWAP (1U << 5)

typedef struct {
    int               uio_fd;
    volatile uint32_t *regs;
} clai_kv260_dev_t;

static inline int clai_kv260_open(clai_kv260_dev_t *dev, const char *uio_path) {
    dev->uio_fd = open(uio_path, O_RDWR | O_SYNC);
    if (dev->uio_fd < 0) return -1;

    void *ptr = mmap(NULL, CLAI_KV260_MAP_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, dev->uio_fd, 0);
    if (ptr == MAP_FAILED) {
        close(dev->uio_fd);
        dev->uio_fd = -1;
        return -2;
    }
    dev->regs = (volatile uint32_t *)ptr;
    if (dev->regs[CLAI_CSR_MAGIC >> 2] != CLAI_KV260_MAGIC) {
        munmap((void *)ptr, CLAI_KV260_MAP_BYTES);
        close(dev->uio_fd);
        return -3;
    }
    return 0;
}

static inline void clai_kv260_close(clai_kv260_dev_t *dev) {
    if (dev->regs) munmap((void *)dev->regs, CLAI_KV260_MAP_BYTES);
    if (dev->uio_fd >= 0) close(dev->uio_fd);
    dev->regs = NULL;
    dev->uio_fd = -1;
}

/* Blocks Zynq PS thread with 0% CPU until Rung 0 asserts `irq_escalate_to_ps`! */
static inline int clai_kv260_wait_escalation_irq(clai_kv260_dev_t *dev, uint32_t *irq_count_out) {
    uint32_t unmask = 1U;
    if (write(dev->uio_fd, &unmask, sizeof(unmask)) != (ssize_t)sizeof(unmask)) return -1;
    if (read(dev->uio_fd, irq_count_out, sizeof(*irq_count_out)) != (ssize_t)sizeof(*irq_count_out)) return -2;
    /* W1C clear pending status in PL */
    dev->regs[CLAI_CSR_STATUS >> 2] = 0x5U;
    return 0;
}

/* Reads hardware-encoded 512-bit EngramBitBus (`[8] uint64_t`) from `0x100..0x13C` */
static inline void clai_kv260_read_latched_bus512(const clai_kv260_dev_t *dev, uint64_t out_words[8]) {
    for (unsigned h = 0; h < 8; ++h) {
        uint32_t lo = dev->regs[(CLAI_CSR_BUS512_BASE >> 2) + h * 2U];
        uint32_t hi = dev->regs[(CLAI_CSR_BUS512_BASE >> 2) + h * 2U + 1U];
        out_words[h] = ((uint64_t)hi << 32) | (uint64_t)lo;
    }
}

/* Programs one LUT6 cell in the inactive Shadow Bank (`~active_bank`) */
static inline void clai_kv260_write_shadow_lut6(
    clai_kv260_dev_t *dev,
    uint8_t layer_sel,
    uint8_t lut_idx,
    uint64_t table_init,
    const uint16_t wires[6]
) {
    uint32_t w_lo = ((uint32_t)(wires[0] & 0x1FFU)) |
                    ((uint32_t)(wires[1] & 0x1FFU) << 9) |
                    ((uint32_t)(wires[2] & 0x1FFU) << 18);
    uint32_t w_hi = ((uint32_t)(wires[3] & 0x1FFU)) |
                    ((uint32_t)(wires[4] & 0x1FFU) << 9) |
                    ((uint32_t)(wires[5] & 0x1FFU) << 18);

    dev->regs[CLAI_CSR_HS_TABLE_LO >> 2] = (uint32_t)(table_init & 0xFFFFFFFFULL);
    dev->regs[CLAI_CSR_HS_TABLE_HI >> 2] = (uint32_t)(table_init >> 32);
    dev->regs[CLAI_CSR_HS_WIRES_LO >> 2] = w_lo;
    dev->regs[CLAI_CSR_HS_WIRES_HI >> 2] = w_hi;
    dev->regs[CLAI_CSR_HS_CTRL >> 2]     = 0x80000000U | ((uint32_t)(layer_sel & 1U) << 8) | (lut_idx & 0x3FU);
}

/* Triggers a single-cycle (4.0 ns @ 250 MHz) glitch-free Shadow -> Active Bank flip */
static inline uint32_t clai_kv260_atomic_hotswap(clai_kv260_dev_t *dev) {
    dev->regs[CLAI_CSR_CTRL >> 2] = CLAI_CTRL_IRQ_ENABLE | CLAI_CTRL_OVERLAY_ENABLE | CLAI_CTRL_ATOMIC_BANK_SWAP;
    return dev->regs[CLAI_CSR_HS_GENERATION >> 2];
}

#ifdef __cplusplus
}
#endif

#endif /* CLAI_KV260_UIO_H */
