"""Talk to the agent without dialing the clinic number from Brazil.

    uv run --env-file .env dial.py +5511999999999            the agent calls this phone
    uv run --env-file .env dial.py --browser +5511999999999  a browser link instead, as if calling from it

Phone: the call goes out through the LiveKit outbound trunk and Twilio, from ROBOT_NUMBER.
The agent is sent into the room only after the phone is answered, so its greeting is not
spoken to a ringing line: expect a second or two of silence after picking up.

Browser: same agent, same tools and database, minus the telephone line (no transfer).
The token sets the sip.phoneNumber attribute, so the agent identifies the caller exactly
as it would from caller ID.
"""

import asyncio
import os
import sys
import urllib.parse
import uuid
from datetime import timedelta

from livekit import api

DISPATCH = api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name="vet-agent")])


async def call_phone(number: str) -> None:
    room = f"call-out-{uuid.uuid4().hex[:8]}"
    async with api.LiveKitAPI() as lk:  # LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
        print(f"ringing {number} from {os.environ['ROBOT_NUMBER']}...")
        await lk.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=os.environ["LIVEKIT_OUTBOUND_TRUNK_ID"],
                sip_call_to=number,
                sip_number=os.environ["ROBOT_NUMBER"],
                room_name=room,
                participant_identity=number,
                wait_until_answered=True,
            )
        )
        await lk.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(agent_name="vet-agent", room=room))
        print(f"answered; vet-agent is joining {room}")


def browser_link(number: str) -> str:
    room = f"call-web-{uuid.uuid4().hex[:8]}"
    token = (
        api.AccessToken()
        .with_identity(number)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .with_attributes({"sip.phoneNumber": number})
        .with_room_config(DISPATCH)  # joining the room sends vet-agent in
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )
    query = urllib.parse.urlencode({"liveKitUrl": os.environ["LIVEKIT_URL"], "token": token})
    return f"https://meet.livekit.io/custom?{query}"


if __name__ == "__main__":
    if sys.argv[1] == "--browser":
        print(browser_link(sys.argv[2]))
    else:
        asyncio.run(call_phone(sys.argv[1]))
