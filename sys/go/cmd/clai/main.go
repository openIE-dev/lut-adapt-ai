//go:build goexperiment.simd

package main

import (
	"encoding/binary"
	"encoding/json/v2"
	"fmt"
	"math"
	"math/bits"
	"runtime"
	"simd"
	"sync"
	"time"
)

const (
	BusWires   = 512
	LutWidth   = 512
	NumClasses = 8
	MambaDim   = 64
)

type TelemetryReceipt struct {
	Toolchain        string  `json:"toolchain"`
	Arch             string  `json:"arch"`
	SimdVectorBits   int     `json:"simd_vector_bits"`
	Episodes         int     `json:"episodes"`
	Rung0LatencyUs   float64 `json:"rung0_latency_us"`
	Rung0Satiated    int     `json:"rung0_satiated"`
	SoinnPrototypes  int     `json:"soinn_prototypes"`
	NovelClassRecall float64 `json:"novel_class_99_recall"`
	JoulesPerCommit  float64 `json:"joules_per_verified_commit"`
}

// EngramStore holds packed uint64 conditional signatures in L3-resident slices.
type EngramStore struct {
	Table [8][65536]uint64
	Salts [8]uint64
}

func NewEngramStore() *EngramStore {
	e := &EngramStore{}
	s := uint64(0xCAFEBABE12345678)
	next := func() uint64 {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		return s
	}
	for h := range 8 {
		e.Salts[h] = next() | 1
		for b := range 65536 {
			e.Table[h][b] = next()
		}
	}
	return e
}

// FetchBitPlanes64 folds 64 variable-length streams into [512] uint64 episode-major bit-planes.
func (e *EngramStore) FetchBitPlanes64(episodes [][]byte, out *[BusWires]uint64) {
	var epWords [64][8]uint64
	for ep := range 64 {
		buf := episodes[ep]
		var mix, x0, x1 uint64
		var slotSum [16]uint32
		for idx, b := range buf {
			slotSum[idx&15] += uint32(b)
		}
		nChunks := uint32((len(buf) + 15) / 16)
		if nChunks == 0 {
			nChunks = 1
		}

		// Bounds-check-eliminated strided loop idiom for Go 1.27 prove pass
		for i := 0; i < len(buf)-4+1; i += 4 {
			w := uint64(binary.LittleEndian.Uint32(buf[i : i+4]))
			mix += w * uint64((i/4+1)*0x9E3779B1)
			if (i & 4) == 0 {
				x0 ^= w
			} else {
				x1 ^= w << 32
			}
		}

		// Words 0..1: clean unscrambled 128-bit streaming XOR fold
		epWords[ep][0] = x0
		epWords[ep][1] = x1

		// Words 2..5: 16-slot x 16-level monotonic thermometer bit-planes
		for grp := range 4 {
			var pack64 uint64
			for s := range 4 {
				norm := slotSum[grp*4+s] / nChunks
				var t16 uint64
				for lvl := range 16 {
					if norm >= uint32((lvl+1)*15) {
						t16 |= 1 << lvl
					}
				}
				pack64 |= t16 << (s * 16)
			}
			epWords[ep][2+grp] = pack64
		}

		// Words 6..7: Associative Engram N-gram lookup
		for h := 6; h < 8; h++ {
			idx := ((mix * e.Salts[h]) ^ (mix >> 13)) & 65535
			epWords[ep][h] = e.Table[h][idx]
		}
	}

	for h := range 8 {
		base := h * 64
		for w := range 64 {
			var col uint64
			for ep := range 64 {
				col |= ((epWords[ep][h] >> w) & 1) << ep
			}
			out[base+w] = col
		}
	}
}

type Lut6Layer struct {
	Wires [LutWidth][6]uint16
	Masks [LutWidth]uint64
}

func NewLut6Layer(seed uint64) *Lut6Layer {
	l := &Lut6Layer{}
	s := seed
	for n := range LutWidth {
		for p := range 6 {
			s = s*6364136223846793005 + 1442695040888963407
			l.Wires[n][p] = uint16((s >> 32) % BusWires)
		}
		s = s*6364136223846793005 + 1442695040888963407
		l.Masks[n] = s
	}
	return l
}

// Eval64 evaluates all 64 episodes across 512 LUT6 gates concurrently across worker goroutines.
func (l *Lut6Layer) Eval64(in *[BusWires]uint64, out *[LutWidth]uint64) {
	workers := min(runtime.GOMAXPROCS(0), 8)
	chunk := LutWidth / workers
	var wg sync.WaitGroup
	wg.Add(workers)

	for w := range workers {
		start, end := w*chunk, (w+1)*chunk
		go func(lo, hi int) {
			defer wg.Done()
			for n := lo; n < hi; n++ {
				idx := l.Wires[n]
				w0, w1, w2 := in[idx[0]], in[idx[1]], in[idx[2]]
				w3, w4, w5 := in[idx[3]], in[idx[4]], in[idx[5]]
				nw0, nw1, nw2, nw3, nw4, nw5 := ^w0, ^w1, ^w2, ^w3, ^w4, ^w5
				t := l.Masks[n]
				var acc uint64
				for m := range uint64(64) {
					tMask := -(t >> m & 1)
					b0 := nw0 ^ ((w0 ^ nw0) & -(m & 1))
					b1 := nw1 ^ ((w1 ^ nw1) & -((m >> 1) & 1))
					b2 := nw2 ^ ((w2 ^ nw2) & -((m >> 2) & 1))
					b3 := nw3 ^ ((w3 ^ nw3) & -((m >> 3) & 1))
					b4 := nw4 ^ ((w4 ^ nw4) & -((m >> 4) & 1))
					b5 := nw5 ^ ((w5 ^ nw5) & -((m >> 5) & 1))
					acc |= tMask & b0 & b1 & b2 & b3 & b4 & b5
				}
				out[n] = acc
			}
		}(start, end)
	}
	wg.Wait()
}

// SimdCosineDot uses Go 1.27's portable hardware `simd` package (ARM64 NEON / AMD64 AVX-512)
// for fused multiply-add vector dot products with zero heap allocations.
func SimdCosineDot(a, b *[MambaDim]float32) (float32, int) {
	var zeroVec simd.Float32s
	lanes := zeroVec.Len()
	acc := simd.BroadcastFloat32s(0)
	sA, sB := a[:], b[:]
	for i := 0; i <= len(sA)-lanes; i += lanes {
		va := simd.LoadFloat32s(sA[i:])
		vb := simd.LoadFloat32s(sB[i:])
		acc = va.MulAdd(vb, acc)
	}
	var tmp [16]float32
	acc.Store(tmp[:lanes])
	var sum float32
	for i := range lanes {
		sum += tmp[i]
	}
	return sum, lanes * 32
}

type HcSoinnRouter struct {
	Prototypes [][MambaDim]float32
	Labels     []int
	Taus       []float32
}

func (g *HcSoinnRouter) Normalize(v *[MambaDim]float32) {
	dot, _ := SimdCosineDot(v, v)
	inv := float32(1.0 / math.Sqrt(float64(dot)+1e-8))
	for i := range MambaDim {
		v[i] *= inv
	}
}

func (g *HcSoinnRouter) Assimilate(z [MambaDim]float32, label int) {
	g.Normalize(&z)
	bestD := float32(2.0)
	bestIdx := -1
	for i := range g.Prototypes {
		dot, _ := SimdCosineDot(&g.Prototypes[i], &z)
		d := max(float32(0), 1.0-dot)
		if d < bestD {
			bestD = d
			bestIdx = i
		}
	}
	if bestIdx >= 0 && g.Labels[bestIdx] == label && bestD <= g.Taus[bestIdx] {
		for j := range MambaDim {
			g.Prototypes[bestIdx][j] = 0.8*g.Prototypes[bestIdx][j] + 0.2*z[j]
		}
		g.Normalize(&g.Prototypes[bestIdx])
		return
	}
	g.Prototypes = append(g.Prototypes, z)
	g.Labels = append(g.Labels, label)
	g.Taus = append(g.Taus, 0.25)
}

func (g *HcSoinnRouter) Query(z [MambaDim]float32, margin float32) (int, bool) {
	if len(g.Prototypes) < 2 {
		return -1, false
	}
	g.Normalize(&z)
	bestD := float32(2.0)
	bestIdx := 0
	for i := range g.Prototypes {
		dot, _ := SimdCosineDot(&g.Prototypes[i], &z)
		d := max(float32(0), 1.0-dot)
		if d < bestD {
			bestD = d
			bestIdx = i
		}
	}
	winLabel := g.Labels[bestIdx]
	rivalD := float32(2.0)
	for i := range g.Prototypes {
		if g.Labels[i] != winLabel {
			dot, _ := SimdCosineDot(&g.Prototypes[i], &z)
			rivalD = min(rivalD, max(float32(0), 1.0-dot))
		}
	}
	if bestD <= g.Taus[bestIdx] && (rivalD-bestD) >= margin {
		return winLabel, true
	}
	return winLabel, false
}

func main() {
	engram := NewEngramStore()
	l0 := NewLut6Layer(11)
	l1 := NewLut6Layer(22)

	episodes := make([][]byte, 64)
	for ep := range 64 {
		b := make([]byte, 20+(ep%40))
		for i := range b {
			b[i] = byte((ep*19 + i*7) & 0xFF)
		}
		episodes[ep] = b
	}

	var p0, p1 [BusWires]uint64
	t0 := time.Now()
	engram.FetchBitPlanes64(episodes, &p0)
	l0.Eval64(&p0, &p1)
	l1.Eval64(&p1, &p0)
	dtR0 := time.Since(t0)

	satiated := 0
	for ep := range 64 {
		mask := uint64(1) << ep
		best, second := 0, 0
		for c := range NumClasses {
			votes := 0
			for k := range LutWidth / NumClasses {
				votes += bits.OnesCount64(p0[c*(LutWidth/NumClasses)+k] & mask)
			}
			if votes > best {
				second = best
				best = votes
			} else if votes > second {
				second = votes
			}
		}
		if best-second >= 3 {
			satiated++
		}
	}

	// Verify SIMD HC-SOINN Novel Class 99 attachment
	soinn := &HcSoinnRouter{}
	var base0, base1, novel99 [MambaDim]float32
	base0[0], base0[1] = 1.0, 0.5
	base1[10], base1[11] = 1.0, -0.5
	novel99[32], novel99[33] = 1.0, 0.8
	_, vecBits := SimdCosineDot(&base0, &base1)

	soinn.Assimilate(base0, 0)
	soinn.Assimilate(base1, 1)
	soinn.Assimilate(novel99, 99)

	hits := 0
	for trial := range 16 {
		q := novel99
		q[(trial*3)%MambaDim] += 0.05
		if pred, ok := soinn.Query(q, 0.10); ok && pred == 99 {
			hits++
		}
	}

	rec := TelemetryReceipt{
		Toolchain:        runtime.Version(),
		Arch:             runtime.GOARCH,
		SimdVectorBits:   vecBits,
		Episodes:         64,
		Rung0LatencyUs:   float64(dtR0.Nanoseconds()) / 1e3,
		Rung0Satiated:    satiated,
		SoinnPrototypes:  len(soinn.Prototypes),
		NovelClassRecall: float64(hits) / 16.0,
		JoulesPerCommit:  (dtR0.Seconds() * 3.0) / 64.0,
	}

	payload, err := json.Marshal(rec)
	if err != nil {
		panic(err)
	}
	fmt.Println(string(payload))
}
