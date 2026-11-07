#!/usr/bin/env python3
"""Patch a scratch copy of syclnn: with SYCLNN_DISCARD set, the in-order queue discards its events.

Not part of the library. Dependencies are dropped, because a discarded event cannot be waited on, and
the in-order queue already orders every command.
  scripts/syclnn-discard-events.py <copy-of-syclnn>
"""
import re
import sys
from pathlib import Path

base = Path(sys.argv[1])/'include/syclnn'
blas = (base/'blas.hpp').read_text()
blas = blas.replace('namespace syclnn {\n', '''namespace syclnn {

inline const bool g_discard = std::getenv("SYCLNN_DISCARD") != nullptr;
inline const std::vector<sycl::event> &filt(const std::vector<sycl::event> &v) {
    static const std::vector<sycl::event> none;
    return g_discard ? none : v;
}
''', 1)
blas = blas.replace('#include <stdexcept>\n', '#include <cstdlib>\n#include <stdexcept>\n', 1)
blas = blas.replace('h.depends_on(deps)', 'h.depends_on(filt(deps))')
blas = re.sub(r'(?<!&)\bdeps\)', 'filt(deps))', blas).replace('filt(filt(deps))', 'filt(deps)')
(base/'blas.hpp').write_text(blas)
net = (base/'network.hpp').read_text()
net = net.replace('h.depends_on(deps)', 'h.depends_on(filt(deps))').replace('h.depends_on(evs)', 'h.depends_on(filt(evs))')
net = net.replace('m_queue.memcpy(dst, src, n * sizeof(T), deps)', 'm_queue.memcpy(dst, src, n * sizeof(T), filt(deps))')
net = net.replace('sycl::event::wait_and_throw(evs);', 'if (g_discard) m_queue.wait_and_throw(); else sycl::event::wait_and_throw(evs);')
net = net.replace('m_prof.timed_wait([&] { e.wait_and_throw(); });', 'm_prof.timed_wait([&] { if (g_discard) m_queue.wait_and_throw(); else e.wait_and_throw(); });')
net = net.replace('return sycl::queue(dev, handler, {sycl::property::queue::in_order{}});\n    return sycl::queue(dev, handler);', '''{
        if (g_discard)
            return sycl::queue(dev, handler, {sycl::property::queue::in_order{}, sycl::ext::oneapi::property::queue::discard_events{}});
        return sycl::queue(dev, handler, {sycl::property::queue::in_order{}});
    }
    return sycl::queue(dev, handler);''', 1)
(base/'network.hpp').write_text(net)
