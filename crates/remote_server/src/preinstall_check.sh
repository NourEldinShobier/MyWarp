#!/usr/bin/env bash
# Pre-install check for the Warp remote-server binary.
#
# stdout prints a structured key=value summary. Exit code 0 means probing completed;
# non-zero means probing failed; the client treats it as `status=unknown` and fails open.
#
# Important: the Warp Linux remote-server is now built by warp_release.yml as a static link for the
# `x86_64-unknown-linux-musl` target (static-musl). The artifact does not depend on the host's
# dynamic libc, so it runs on any Linux x86_64 host - including old glibc
# distributions (CentOS 7 = 2.17, Amazon Linux 2 = 2.26, Ubuntu 20.04 / Debian 11
# = 2.31) and musl distributions (Alpine, etc.).
#
# Since the binary is static, libc probing is no longer used as a "gate"; it is kept only as telemetry.

set -u

# Legacy field: required_glibc is kept for compatibility with older clients' parsing logic.
# A static musl binary actually has no glibc lower bound; this is output only for backward compatibility
# and no longer participates in the status decision below.
required_glibc="2.17"
echo "required_glibc=${required_glibc}"

# 1. Identify the libc family and, for glibc, its version (telemetry only, does not affect status).
libc_family="unknown"
libc_version=""

if version=$(getconf GNU_LIBC_VERSION 2>/dev/null); then
    # Output looks like: "glibc 2.35"
    libc_family="glibc"
    libc_version="${version##* }"
elif ldd_out=$(ldd --version 2>&1 | head -n1); then
    case "$ldd_out" in
        *musl*)   libc_family="musl"   ;;
        *uClibc*) libc_family="uclibc" ;;
        *)
            v=$(printf '%s\n' "$ldd_out" | grep -oE '[0-9]+\.[0-9]+' | head -n1)
            if [ -n "$v" ]; then
                libc_family="glibc"
                libc_version="$v"
            fi
            ;;
    esac
fi

echo "libc_family=${libc_family}"
[ -n "$libc_version" ] && echo "libc_version=${libc_version}"

# 2. Determine the support status.
#
# remote-server is a static musl binary that does not link the host libc, so any glibc version
# (including below 2.35) and musl / uclibc hosts can run it. As long as this is identified as a
# Linux x86_64 host, report `supported`; if no libc clue is found at all (not even
# getconf or ldd), fall back to `unknown` so the client fails open and tries to install as usual.
status="unknown"
reason=""

if [ "$libc_family" = "glibc" ] \
   || [ "$libc_family" = "musl" ] \
   || [ "$libc_family" = "uclibc" ] \
   || [ "$libc_family" = "bionic" ]; then
    status="supported"
fi

echo "status=${status}"
if [ -n "$reason" ]; then
    echo "reason=${reason}"
fi
