import os
import argparse
from livekit import api
from dotenv import load_dotenv

load_dotenv()

def generate_token(room_name: str, participant_identity: str) -> str:
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in environment variables.")

    token = api.AccessToken(api_key, api_secret) \
        .with_identity(participant_identity) \
        .with_name(participant_identity) \
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
    
    return token.to_jwt()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LiveKit Access Token for Client App")
    parser.add_argument("--room", default="lelamp-room", help="Room name (default: lelamp-room)")
    parser.add_argument("--user", default="user-app", help="Participant identity (default: user-app)")
    
    args = parser.parse_args()
    
    try:
        jwt = generate_token(args.room, args.user)
        print(f"Room: {args.room}")
        print(f"User: {args.user}")
        print("-" * 20)
        print(f"Token:\n{jwt}")
        print("-" * 20)
        print("Use this token in the Web Client to connect.")
    except Exception as e:
        print(f"Error: {e}")
