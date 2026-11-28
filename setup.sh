#!/bin/bash
# Run every 10 minutes by /opt/polybot/bootstrap.sh (as root). Safe to run again and again.
set -e
REPO=/opt/polybot/repo; VENV=/opt/polybot/venv; STATE=/opt/polybot/state
mkdir -p "$STATE" /opt/polybot/data
cd "$REPO"
git config user.name "polybot-server"; git config user.email "polybot-server@users.noreply.github.com"
git pull -q --rebase --autostash || true
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
REQ=$(sha256sum requirements.txt | cut -c1-16)
if [ "$(cat $STATE/req 2>/dev/null)" != "$REQ" ]; then
  "$VENV/bin/pip" install -q --upgrade pip && "$VENV/bin/pip" install -q -r requirements.txt && echo "$REQ" > $STATE/req
fi
UNIT=/etc/systemd/system/polybot.service
cat > /tmp/polybot.service <<UNITEOF
[Unit]
Description=Polybot paper-trading research bot (no real money)
After=network-online.target
Wants=network-online.target
[Service]
WorkingDirectory=$REPO
Environment=POLYBOT_DATA=/opt/polybot/data
ExecStart=$VENV/bin/python -u bot.py
Restart=always
RestartSec=15
MemoryMax=700M
[Install]
WantedBy=multi-user.target
UNITEOF
if ! cmp -s /tmp/polybot.service $UNIT; then cp /tmp/polybot.service $UNIT; systemctl daemon-reload; systemctl enable -q polybot; fi
CODE=$(cat *.py requirements.txt model/* 2>/dev/null | sha256sum | cut -c1-16)
if [ "$(cat $STATE/code 2>/dev/null)" != "$CODE" ] || ! systemctl is-active -q polybot; then
  systemctl restart polybot && echo "$CODE" > $STATE/code
fi

# ---- ALT/BTC paper bot (separate service, same venv) ----
mkdir -p /opt/polybot/altdata
cat > /tmp/altbot.service <<UNITEOF
[Unit]
Description=ALT/BTC paper-trading research bot (no real money)
After=network-online.target
Wants=network-online.target
[Service]
WorkingDirectory=$REPO/altbot
Environment=ALTBOT_DATA=/opt/polybot/altdata
ExecStart=$VENV/bin/python -u altbot.py
Restart=always
RestartSec=15
MemoryMax=250M
[Install]
WantedBy=multi-user.target
UNITEOF
AUNIT=/etc/systemd/system/altbot.service
if [ -f altbot/model/alt15.joblib ]; then
  if ! cmp -s /tmp/altbot.service $AUNIT; then cp /tmp/altbot.service $AUNIT; systemctl daemon-reload; systemctl enable -q altbot; fi
  ACODE=$(cat altbot/*.py altbot/model/* requirements.txt 2>/dev/null | sha256sum | cut -c1-16)
  if [ "$(cat $STATE/altcode 2>/dev/null)" != "$ACODE" ] || ! systemctl is-active -q altbot; then
    systemctl restart altbot && echo "$ACODE" > $STATE/altcode
  fi
fi
