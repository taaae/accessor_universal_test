# Experiment 019 context: FP6 storage and access strategies

## Scope

The next experiment focuses only on six-bit formats. Do not expand to every
other width yet. The purpose is to establish a clean access-strategy taxonomy
and determine whether exact dense six-bit storage is actually worthwhile.

For a sign/exponent/mantissa layout with six total bits, the complete layout set
is E0M5, E1M4, E2M3, E3M2, E4M1, and E5M0. The access experiment should share
the same machinery across these layouts; conversion strategies may still be
specialized where their exponent/mantissa structure requires it.

## Main decision

Padded and dense representations must remain separate result groups. If the
best dense six-bit implementation is slower than an eight-bit padded
representation, the storage saving may not justify using a six-bit format. In
that case, a larger byte-aligned format may provide both simpler access and
better numerical range or precision.

This comparison must use complete DOT/GEMV kernels as well as isolated access
measurements. Dense storage reduces requested bytes, but additional extraction,
cross-word handling, shuffles, register pressure, or synchronization can erase
that advantage.

## Terminology

Avoid using the single word "packed" for several unrelated ideas:

- `padded` versus `dense` describes the storage layout;
- `scalar`, `thread_packet`, and `cooperative` describe the access method;
- `packet_values` describes how many logical values one operation returns;
- `shuffle` or `shared` describes cooperative redistribution;
- native vector arithmetic is a separate arithmetic property.

## Strategy groups

### 1a. Padded scalar accessor

- Store one FP6 code in an eight-bit byte; the unused two bits are fixed zeros.
- Keep the original kernel call site: `value = arr[i]`.
- Each thread independently loads and decodes one value.
- This is the simple, byte-addressable performance baseline, but consumes eight
  storage bits per logical value.

### 1b. Dense scalar accessor

- Store exactly six bits per logical value in a contiguous bitstream.
- Keep the same `value = arr[i]` call-site semantic.
- The accessor computes `bit_offset = 6 * i`, loads the containing word or
  words, and extracts the code independently.
- No shuffle, shared memory, or inter-thread participation is permitted.
- Some adjacent threads may issue overlapping underlying word loads. This is a
  cost of retaining arbitrary scalar indexing over a dense bitstream.

### 2a. Padded per-thread packet

- Keep one eight-bit container per FP6 value.
- One thread loads and processes x2/x4/x8 consecutive values through an API
  such as `arr.load_packet<L>(i)`.
- This isolates thread coarsening, wider byte loads, independent accumulators,
  and loop/address amortization without dense-bit extraction.

### 2b. Dense per-thread packet

- Store exactly six bits per value.
- One thread loads an aligned bit packet and extracts several consecutive FP6
  codes, with no inter-thread communication.
- Compare directly with 2a at equal logical packet width. Their difference
  measures the benefit of reduced storage traffic against dense unpacking cost.

### 3. Dense cooperative packet

- A cooperating thread group loads aligned words from the dense bitstream and
  redistributes codes with warp shuffles.
- The first concrete FP6 packet is:

  ```text
  16 FP6 values
      = 16 * 6 bits
      = 96 dense bits
      = three aligned 32-bit word loads by three loader lanes
      -> shuffle redistribution
      -> candidate consumer mappings:
           16 lanes * 1 value
            8 lanes * 2 values
            4 lanes * 4 values
  ```

- This requires a group/packet accessor API and kernel indexing designed around
  collective participation. It cannot transparently implement arbitrary
  independent `arr[i]` calls.
- Test the consumer mappings separately because fewer consumers increase work
  per thread and register use, while more consumers require more redistribution.

A padded cooperative version is not a primary group: byte-addressable padded
values do not need cross-thread reconstruction. If later required as a control,
label it explicitly rather than mixing it with dense cooperative results.

## Required metadata

Record these dimensions independently instead of encoding them only in a
strategy name:

```text
storage_layout       = padded | dense
access_method        = scalar | thread_packet | cooperative
packet_values        = 1 | 2 | 4 | 8 | 16 | ...
load_word_bits       = 8 | 16 | 32 | 64 | 128
loader_threads       = 1 | 3 | ...
consumer_threads     = 1 | 4 | 8 | 16 | ...
values_per_consumer  = 1 | 2 | 4 | 8 | ...
redistribution       = none | shuffle | shared
arithmetic_type      = fp32 | fp64 | ...
```

This makes it possible to attribute a result to storage density, per-thread
coarsening, load width, or cooperative redistribution rather than placing every
non-scalar method in one unexplained "weird strategies" group.

## Comparisons that must remain visible

1. 1a versus 1b: is dense storage useful while preserving `arr[i]` semantics?
2. 2a versus 2b: does reduced traffic repay dense unpacking at the same
   per-thread packet width?
3. 1b versus 2b: how much does per-thread packet access improve a dense layout?
4. 2b versus 3: does cooperative loading remove enough redundant extraction or
   load work to repay shuffle and synchronization costs?
5. Best padded strategy versus best dense strategy: should six-bit storage be
   used at all instead of an eight-bit container/format?

Accuracy remains a separate property of each E/M layout. A performance win does
not make one six-bit layout numerically interchangeable with another.
