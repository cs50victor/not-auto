#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyjwt",
#     "python-dotenv",
# ]
# ///

import jwt
import time
import os
import sys
from dotenv import load_dotenv

load_dotenv('.env.local', override=True)


api_key = os.environ["LIVEKIT_API_KEY"]
api_secret = os.environ["LIVEKIT_API_SECRET"]
livekit_url = os.environ["LIVEKIT_URL"]
room_name = os.environ.get("LIVEKIT_ROOM_NAME", "bevy_render_room")

if not api_key or not api_secret:
    print("Error: LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env.local")
    sys.exit(1)

# Convert https:// to wss:// if needed
if livekit_url.startswith("https://"):
    livekit_url = livekit_url.replace("https://", "wss://")
elif livekit_url.startswith("http://"):
    livekit_url = livekit_url.replace("http://", "ws://")

payload = {
    "exp": int(time.time()) + 86400,  # 24 hours from now
    "iss": api_key,
    "nbf": int(time.time()) - 5,
    "sub": "viewer",
    "name": "Web Viewer",
    "video": {
        "room": room_name,
        "roomJoin": True,
        "canSubscribe": True,
        "canPublish": False,
        "canPublishData": False
    },
    "iat": int(time.time()),
    "jti": "viewer-" + str(int(time.time()))
}

token = jwt.encode(payload, api_secret, algorithm="HS256")
print("\n" + "="*60)
print("LIVEKIT VIEWER TOKEN")
print("="*60)
print(f"\nRoom: {room_name}")
print(f"LiveKit URL: {livekit_url}")
print(f"\nToken:\n{token}")
print("\n" + "="*60)
print("HOW TO USE:")
print("="*60)
print("\n1. Go to: https://meet.livekit.io/")
print("2. Click 'Custom' tab")
print(f"3. Enter LiveKit URL: {livekit_url}")
print("4. Paste the token above")
print("5. Click 'Connect'")
print("\nYou should see the bevy_spatial_renderer stream!")
print("="*60)