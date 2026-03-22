from __future__ import annotations


def normalize_command(command: str | None) -> str:
    if not command:
        return "follow_lane"
    command = str(command).strip().lower()
    aliases = {
        "left": "turn_left",
        "right": "turn_right",
        "straight": "go_straight",
        "lanefollow": "follow_lane",
    }
    return aliases.get(command, command)


def build_commentary(command: str, speed_kmh: float, brake: float, throttle: float) -> str:
    command = normalize_command(command)
    if brake > 0.25:
        return "The vehicle is slowing down or stopping to stay safe in the current traffic scene."
    if command == "turn_left":
        return "The vehicle is approaching a left turn and should reduce speed while steering left."
    if command == "turn_right":
        return "The vehicle is preparing for a right turn and should keep a controlled speed."
    if command == "go_straight":
        return "The road ahead is drivable and the vehicle should continue straight with stable control."
    if throttle > 0.3 and speed_kmh < 10:
        return "The vehicle is accelerating gently to continue along the route."
    return "The vehicle should follow the lane smoothly while maintaining a safe and stable speed."


def build_prompt(command: str, speed_kmh: float) -> str:
    command = normalize_command(command)
    return (
        "You are an autonomous driving assistant. "
        "Observe the front-view image, vehicle speed, and navigation command. "
        "First produce a short driving commentary, then predict steer, throttle, and brake values.\n"
        f"Speed: {speed_kmh:.2f} km/h\n"
        f"Command: {command}\n"
        "Output format:\n"
        "Commentary: <one sentence>\n"
        "Steer: <float>\n"
        "Throttle: <float>\n"
        "Brake: <float>"
    )


def build_target_text(commentary: str, steer: float, throttle: float, brake: float) -> str:
    return (
        f"Commentary: {commentary}\n"
        f"Steer: {steer:.4f}\n"
        f"Throttle: {throttle:.4f}\n"
        f"Brake: {brake:.4f}"
    )
