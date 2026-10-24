"""Samsung Health's predefined exercise types, from the Samsung Health SDK reference (``EXERCISE_TYPE``).

The names are shortened to the activity itself ("Swimming", not "Swimming, general, leisurely, not lap
swimming"), which is also how most public diabetes datasets label workouts.
"""

from __future__ import annotations

EXERCISE_TYPES: dict[int, str] = {
    0: "Custom",
    1001: "Walking", 1002: "Running",
    2001: "Baseball", 2002: "Softball", 2003: "Cricket",
    3001: "Golf", 3002: "Billiards", 3003: "Bowling",
    4001: "Hockey", 4002: "Rugby", 4003: "Basketball", 4004: "Football", 4005: "Handball",
    4006: "American football",
    5001: "Volleyball", 5002: "Beach volleyball",
    6001: "Squash", 6002: "Tennis", 6003: "Badminton", 6004: "Table tennis", 6005: "Racquetball",
    7001: "Tai chi", 7002: "Boxing", 7003: "Martial arts",
    8001: "Ballet", 8002: "Dancing", 8003: "Ballroom dancing",
    9001: "Pilates", 9002: "Yoga",
    10001: "Stretching", 10002: "Jump rope", 10003: "Hula-hooping", 10004: "Push-ups",
    10005: "Pull-ups", 10006: "Sit-ups", 10007: "Circuit training", 10008: "Mountain climbers",
    10009: "Jumping jacks", 10010: "Burpee", 10011: "Bench press", 10012: "Squats", 10013: "Lunges",
    10014: "Leg presses", 10015: "Leg extensions", 10016: "Leg curls", 10017: "Back extensions",
    10018: "Lat pull-downs", 10019: "Deadlifts", 10020: "Shoulder presses", 10021: "Front raises",
    10022: "Lateral raises", 10023: "Crunches", 10024: "Leg raises", 10025: "Plank",
    10026: "Arm curls", 10027: "Arm extensions",
    11001: "Inline skating", 11002: "Hang gliding", 11003: "Pistol shooting", 11004: "Archery",
    11005: "Horseback riding", 11007: "Cycling", 11008: "Flying disc", 11009: "Roller skating",
    12001: "Aerobics",
    13001: "Hiking", 13002: "Rock climbing", 13003: "Backpacking", 13004: "Mountain biking",
    13005: "Orienteering",
    14001: "Swimming", 14002: "Aquarobics", 14003: "Canoeing", 14004: "Sailing", 14005: "Scuba diving",
    14006: "Snorkeling", 14007: "Kayaking", 14008: "Kitesurfing", 14009: "Rafting",
    14010: "Rowing machine", 14011: "Windsurfing", 14012: "Yachting", 14013: "Water skiing",
    15001: "Step machine", 15002: "Weight machine", 15003: "Exercise bike", 15004: "Rowing machine",
    15005: "Treadmill", 15006: "Elliptical trainer",
    16001: "Cross-country skiing", 16002: "Skiing", 16003: "Ice dancing", 16004: "Ice skating",
    16006: "Ice hockey", 16007: "Snowboarding", 16008: "Alpine skiing", 16009: "Snowshoeing",
}


def exercise_name(code: int | None) -> str | None:
    """The activity's name, or ``Type <code>`` for a code the SDK reference does not list."""
    if code is None:
        return None
    return EXERCISE_TYPES.get(code, f"Type {code}")
