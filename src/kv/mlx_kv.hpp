#pragma once
#include <mlx/mlx.h>
#include <algorithm>
#include <map>
#include <tuple>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tqllm::mlx_backend {
// Chronological [1, heads, rows, dim] storage, with independent state per visit.
// Rows live in a zero-initialised buffer written in place (slice_update), so a step copies only its new rows.
// The row after the last written one is always zero: append() returns the visible rows plus that zero row,
// which attention uses as the virtual sink key without concatenating.
class MlxKvCache {
public:
    explicit MlxKvCache(int context, int window = 0) : capacity_(context), window_(window) {
        if (context <= 0 || window < 0) throw std::invalid_argument("MLX KV capacity");
    }
    int position() const { return position_; }
    int start() const { return start_; }
    void reset() { key_.reset(); value_.reset(); position_ = start_ = begin_ = end_ = rows_ = 0; }
    // Returns {keys, values} of shape [1, heads, visible + 1, dim]; the last row is zero.
    std::pair<mlx::core::array, mlx::core::array> append(const mlx::core::array& k, const mlx::core::array& v) {
        namespace mx = mlx::core;
        if (k.ndim() != 4 || k.shape() != v.shape() || k.dtype() != v.dtype() || k.shape(0) != 1 ||
            k.shape(2) <= 0 || position_ + k.shape(2) > capacity_ ||
            (key_ && (key_->shape(1) != k.shape(1) || key_->shape(3) != k.shape(3) || key_->dtype() != k.dtype())))
            throw std::invalid_argument("MLX KV append shape/capacity");
        const int n = k.shape(2);
        if (!key_ || end_ + n + 1 > rows_) reserve(k, n);
        const int heads = k.shape(1), dim = k.shape(3);
        key_ = mx::slice_update(*key_, k, {0, 0, end_, 0}, {1, heads, end_ + n, dim});
        value_ = mx::slice_update(*value_, v, {0, 0, end_, 0}, {1, heads, end_ + n, dim});
        end_ += n;
        position_ += n;
        // Keep all rows needed by this chunk's earliest query; trim on retain_window().
        return {mx::slice(*key_, {0, 0, begin_, 0}, {1, heads, end_ + 1, dim}),
                mx::slice(*value_, {0, 0, begin_, 0}, {1, heads, end_ + 1, dim})};
    }
    void retain_window() {
        if (!window_ || end_ - begin_ <= window_) return;
        begin_ = end_ - window_;
        start_ = position_ - window_;
    }
    // Speculative decoding: drop the rows after `position` (a rejected verify block). The dropped rows are zeroed
    // so the row after the last kept one is the zero sink key again. Run retain_window() after it, not before:
    // the kept window must still be in the buffer.
    void truncate(int position) {
        namespace mx = mlx::core;
        if (position < start_ || position > position_) throw std::invalid_argument("MLX KV truncate position");
        const int drop = position_ - position;
        if (!drop) return;
        const int heads = key_->shape(1), dim = key_->shape(3);
        const auto& zero = zero_rows(heads, drop, dim, key_->dtype());
        key_ = mx::slice_update(*key_, zero, {0, 0, end_ - drop, 0}, {1, heads, end_, dim});
        value_ = mx::slice_update(*value_, zero, {0, 0, end_ - drop, 0}, {1, heads, end_, dim});
        end_ -= drop;
        position_ = position;
    }
    // Visible rows followed by k / v [1, heads, n, dim], without keeping k / v: the next append overwrites them.
    // It writes past the last row, so only for caches that are not read through append()'s zero sink row.
    std::pair<mlx::core::array, mlx::core::array> peek(const mlx::core::array& k, const mlx::core::array& v) {
        namespace mx = mlx::core;
        if (!key_) return {k, v};
        const int n = k.shape(2), heads = k.shape(1), dim = k.shape(3);
        if (end_ + n + 1 > rows_) reserve(k, n);
        key_ = mx::slice_update(*key_, k, {0, 0, end_, 0}, {1, heads, end_ + n, dim});
        value_ = mx::slice_update(*value_, v, {0, 0, end_, 0}, {1, heads, end_ + n, dim});
        return {mx::slice(*key_, {0, 0, begin_, 0}, {1, heads, end_ + n, dim}),
                mx::slice(*value_, {0, 0, begin_, 0}, {1, heads, end_ + n, dim})};
    }
    void collect(std::vector<mlx::core::array>& out) const {
        if (key_) { out.push_back(*key_); out.push_back(*value_); }
    }
    void settle() const {
        if (key_) mlx::core::eval({*key_, *value_});
    }
private:
    // Zero rows for truncate(), evaluated once per shape (a verify block drops 1..7 rows each round): no fill kernel per call.
    static const mlx::core::array& zero_rows(int heads, int rows, int dim, mlx::core::Dtype dtype) {
        static std::map<std::tuple<int, int, int, int>, mlx::core::array> cache;
        const auto key = std::make_tuple(heads, rows, dim, int(dtype.val()));
        auto it = cache.find(key);
        if (it == cache.end()) {
            auto z = mlx::core::zeros({1, heads, rows, dim}, dtype);
            mlx::core::eval(z);
            it = cache.emplace(key, z).first;
        }
        return it->second;
    }
    // New zero buffer holding the kept rows at the front and room for n more rows plus the zero row.
    void reserve(const mlx::core::array& like, int n) {
        namespace mx = mlx::core;
        const int kept = end_ - begin_, heads = like.shape(1), dim = like.shape(3);
        const int need = kept + n + 1;
        // Full attention doubles (amortised copy), and takes the whole capacity once a doubling passes half of it: else
        // the last doubling can fall just short of the capacity and decode then copies the whole cache again (5-7 % slower
        // decode at 128K-256K after 4096-row prompt chunks). A sliding window compacts about once per window of steps; a
        // chunk wider than the window reserves only what it needs (it re-reserves on every append anyway, and a buffer
        // sized for it would stay that large for decode).
        const int grow = std::max(2 * rows_, 256);
        const int rows = window_ ? (n > window_ ? need : need + window_)
                                 : std::max(need, grow > (capacity_ + 1) / 2 ? capacity_ + 1 : grow);
        auto fresh_k = mx::zeros({1, heads, rows, dim}, like.dtype());
        auto fresh_v = mx::zeros({1, heads, rows, dim}, like.dtype());
        if (key_ && kept) {
            fresh_k = mx::slice_update(fresh_k, mx::slice(*key_, {0, 0, begin_, 0}, {1, heads, end_, dim}),
                                       {0, 0, 0, 0}, {1, heads, kept, dim});
            fresh_v = mx::slice_update(fresh_v, mx::slice(*value_, {0, 0, begin_, 0}, {1, heads, end_, dim}),
                                       {0, 0, 0, 0}, {1, heads, kept, dim});
        }
        key_ = fresh_k; value_ = fresh_v;
        begin_ = 0; end_ = kept; rows_ = rows;
    }
    int capacity_, window_, position_ = 0, start_ = 0;
    int begin_ = 0, end_ = 0, rows_ = 0;  // buffer rows [begin_, end_) hold positions [start_, position_)
    std::optional<mlx::core::array> key_, value_;
};
}  // namespace tqllm::mlx_backend
