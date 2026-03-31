import argparse


def setup_motors(port: str, lamp_id: str):
    from lelamp.follower.lelamp_follower import LeLampFollower, LeLampFollowerConfig

    config = LeLampFollowerConfig(
        port=port,
        id=lamp_id,
    )
    leader = LeLampFollower(config)
    leader.setup_motors()


def main():
    parser = argparse.ArgumentParser(description="Setup motors for LeLamp follower")
    parser.add_argument('--id', type=str, required=True, help='ID of the lamp')
    parser.add_argument('--port', type=str, required=True, help='Serial port for the lamp')
    args = parser.parse_args()

    setup_motors(args.port, args.id)

if __name__ == "__main__":
    main()
