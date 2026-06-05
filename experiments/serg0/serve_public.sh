#!/usr/bin/env bash
# Serve a directory on the public internet, robustly: a local http.server + a public tunnel,
# cloudflared (clean, no interstitial) preferred, localtunnel (proven) as fallback. ONLY
# announces a URL after it actually returns 200 (cloudflared quick tunnels need warmup; a
# registered tunnel is not yet a reachable one). Long-lived — leave it running.
#
# Usage: serve_public.sh <dir> [local_port]    # writes the chosen URL to <dir>/.public_url
set -uo pipefail
DIR=${1:?usage: serve_public.sh <dir> [port]}
PORT=${2:-8088}
CF="$HOME/bin/cloudflared"
URLFILE="$DIR/.public_url"
rm -f "$URLFILE"

python3 -m http.server "$PORT" --bind 0.0.0.0 --directory "$DIR" >"/tmp/serve_$PORT.log" 2>&1 &
SRV=$!
echo "local server pid=$SRV on 0.0.0.0:$PORT serving $DIR"
sleep 2

verify() {  # verify <url> [header]; polls up to ~90s for a 200
    local url="$1"
    shift
    local i code
    for i in $(seq 1 30); do
        if [ $# -gt 0 ]; then
            code=$(curl -s -o /dev/null -m 10 -H "$1" -w "%{http_code}" "$url/" 2>/dev/null || echo 000)
        else
            code=$(curl -s -o /dev/null -m 10 -w "%{http_code}" "$url/" 2>/dev/null || echo 000)
        fi
        [ "$code" = "200" ] && return 0
        sleep 3
    done
    return 1
}

PUBURL=""
TUN=""

# --- cloudflared first (clean URL, no interstitial) ---
if [ -x "$CF" ]; then
    echo "trying cloudflared..."
    "$CF" tunnel --url "http://localhost:$PORT" --no-autoupdate >"/tmp/cf_$PORT.log" 2>&1 &
    TUN=$!
    for i in $(seq 1 30); do
        PUBURL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "/tmp/cf_$PORT.log" | head -1)
        [ -n "$PUBURL" ] && break
        sleep 1
    done
    if [ -n "$PUBURL" ] && verify "$PUBURL"; then
        echo "PUBLIC (cloudflared): $PUBURL"
        printf '%s\n' "$PUBURL" >"$URLFILE"
    else
        echo "cloudflared not reachable in time — falling back to localtunnel"
        [ -n "$TUN" ] && kill "$TUN" 2>/dev/null
        PUBURL=""
    fi
fi

# --- localtunnel fallback (proven; browser shows an interstitial wanting the password) ---
if [ -z "$PUBURL" ]; then
    echo "trying localtunnel..."
    npx -y localtunnel --port "$PORT" >"/tmp/lt_$PORT.log" 2>&1 &
    TUN=$!
    for i in $(seq 1 40); do
        PUBURL=$(grep -oE 'https://[a-z0-9-]+\.loca\.lt' "/tmp/lt_$PORT.log" | head -1)
        [ -n "$PUBURL" ] && break
        sleep 1
    done
    PW=$(curl -s -m 10 https://loca.lt/mytunnelpassword 2>/dev/null || echo "?")
    if [ -n "$PUBURL" ] && verify "$PUBURL" "bypass-tunnel-reminder: 1"; then
        echo "PUBLIC (localtunnel): $PUBURL"
        echo "  browser interstitial password (= this box's public IP): $PW"
        printf '%s\n' "$PUBURL" >"$URLFILE"
        printf 'LT_PASSWORD=%s\n' "$PW" >>"$URLFILE"
    else
        echo "ERROR: no public tunnel could be established (cloudflared + localtunnel both failed)"
        [ -n "$TUN" ] && kill "$TUN" 2>/dev/null
        kill "$SRV" 2>/dev/null
        exit 1
    fi
fi

echo "server pid=$SRV tunnel pid=$TUN url-file=$URLFILE — leave running; kill to stop."
wait
