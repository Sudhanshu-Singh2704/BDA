from collections import deque

class DGIM:
    def __init__(self, window_size):
        self.N = window_size
        self.buckets = deque()  # stores (size, timestamp)
        self.time = 0

    def _remove_old_buckets(self):
        while self.buckets and self.buckets[0][1] <= self.time - self.N:
            self.buckets.popleft()

    def _merge_buckets(self):
        sizes = {}
        i = len(self.buckets) - 1
        while i >= 0:
            size, timestamp = self.buckets[i]
            sizes[size] = sizes.get(size, 0) + 1
            if sizes[size] > 2:
                # merge the two oldest of this size: buckets[i] and buckets[i+1]
                newer_bucket = self.buckets[i + 1]
                older_bucket = self.buckets[i]
                new_size = size * 2
                new_timestamp = newer_bucket[1]

                # remove those exact tuples and insert merged
                self.buckets.remove(older_bucket)
                self.buckets.remove(newer_bucket)
                self.buckets.insert(i, (new_size, new_timestamp))

                sizes.clear()
                i = len(self.buckets)
            i -= 1

    def process_bit(self, bit):
        self.time += 1
        if bit == 1:
            self.buckets.append((1, self.time))
            self._merge_buckets()
        self._remove_old_buckets()

    def count_ones(self):
        total = 0
        last_timestamp = self.time - self.N
        for i, (size, timestamp) in enumerate(reversed(self.buckets)):
            if i == 0:
                total += size
            elif timestamp <= last_timestamp:
                total += size / 2
                break
            else:
                total += size
        return int(total)

    def print_buckets_inline(self):
        if not self.buckets:
            print("  Buckets: None")
            return
        bucket_str = " | ".join([f"(s={s},ts={t})" for s, t in self.buckets])
        print(f"  Buckets: {bucket_str}")


def parse_bits(raw_input: str):
    """Robust parsing: collect all characters '0' or '1' in order.
       If none found, try splitting on whitespace and converting tokens to ints.
    """
    raw = raw_input.strip()
    bits = [int(ch) for ch in raw if ch in "01"]
    if bits:
        return bits
    # fallback: whitespace tokens (e.g., "1 0 1")
    try:
        tokens = raw.replace(",", " ").split()
        return [int(t) for t in tokens]
    except Exception:
        raise ValueError("Couldn't parse bits. Enter only 0/1 (space/comma separated or contiguous).")


if __name__ == "__main__":
    raw = input("Enter bit stream (examples: '1 0 1 1', '100101', '1,0,1'): ")
    bits = parse_bits(raw)
    N = int(input("Enter window size N: "))

    dgim = DGIM(window_size=N)
    print("\n--- DGIM Execution Trace ---\n")
    for bit in bits:
        dgim.process_bit(bit)
        count = dgim.count_ones()
        print(f"t={dgim.time:02d} | bit={bit} | Approx 1’s={count}", end="  ")
        dgim.print_buckets_inline()
