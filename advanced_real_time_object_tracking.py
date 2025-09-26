

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class SyntheticObject:
    """Data structure representing a synthetic object in the scene."""

    id: int
    x: float
    y: float
    w: int
    h: int
    dx: float
    dy: float
    color: Tuple[int, int, int]
    shape: str = "rect"  # either 'rect' or 'circle'
    trajectory: List[Tuple[int, int]] = field(default_factory=list)

    def update_position(self, frame_width: int, frame_height: int) -> None:
        """Update the object's position, bouncing off the frame edges."""
        next_x = self.x + self.dx
        next_y = self.y + self.dy

        # bounce horizontally
        if next_x < 0 or next_x > (frame_width - self.w):
            self.dx = -self.dx
            next_x = self.x + self.dx

        # bounce vertically
        if next_y < 0 or next_y > (frame_height - self.h):
            self.dy = -self.dy
            next_y = self.y + self.dy

        self.x = next_x
        self.y = next_y

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Return integer bounding box coordinates (x, y, w, h)."""
        return int(self.x), int(self.y), self.w, self.h

    def center(self) -> Tuple[float, float]:
        """Return the centre of the bounding box."""
        cx = self.x + self.w / 2.0
        cy = self.y + self.h / 2.0
        return cx, cy


@dataclass
class Occluder:
    """A simple rectangle used to occlude parts of the frame."""

    x: float
    y: float
    w: int
    h: int
    dx: float
    dy: float
    color: Tuple[int, int, int] = (0, 0, 0)  # default black

    def update_position(self, frame_width: int, frame_height: int) -> None:
        """Move the occluder across the frame, bouncing off edges."""
        next_x = self.x + self.dx
        next_y = self.y + self.dy

        if next_x < 0 or next_x > (frame_width - self.w):
            self.dx = -self.dx
            next_x = self.x + self.dx

        if next_y < 0 or next_y > (frame_height - self.h):
            self.dy = -self.dy
            next_y = self.y + self.dy

        self.x = next_x
        self.y = next_y

    def bounding_box(self) -> Tuple[int, int, int, int]:
        return int(self.x), int(self.y), self.w, self.h


def create_tracker(tracker_type: str):
    
    ttype = tracker_type.upper()
    if ttype == "MIL":
        if hasattr(cv2, "TrackerMIL_create"):
            return cv2.TrackerMIL_create()
        raise ValueError("MIL tracker is not available in your OpenCV build.")
    elif ttype == "GOTURN":
        # GOTURN requires external caffe model files. Inform the user
        raise ValueError(
            "GOTURN tracker requires pre‑trained model files and cannot be used "
            "in this environment without providing the necessary weights."
        )
    else:
        raise ValueError(
            f"Tracker type '{tracker_type}' is not supported. Currently only 'MIL' "
            "is supported in this environment."
        )


def create_kalman_filter(initial_state: Tuple[float, float, float, float]) -> cv2.KalmanFilter:
   
    kf = cv2.KalmanFilter(4, 2)
    dt = 1.0  # time step between frames
    # State transition matrix (constant velocity)
    kf.transitionMatrix = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    # Measurement matrix: we observe x and y only
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
    )
    # Process noise covariance: tune to reflect expected acceleration
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    # Measurement noise covariance: tune to reflect sensor noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    # Posterior error covariance
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    # Initial state estimate
    kf.statePost = np.array(initial_state, dtype=np.float32).reshape(4, 1)
    return kf


def generate_synthetic_objects(
    num_objects: int,
    frame_width: int,
    frame_height: int,
    min_size: int,
    max_size: int,
    min_speed: float,
    max_speed: float,
    random_shapes: bool,
) -> List[SyntheticObject]:
    """Generate a list of synthetic objects with random properties."""
    objects = []
    rng = np.random.default_rng()
    for i in range(num_objects):
        w = int(rng.integers(min_size, max_size + 1))
        h = int(rng.integers(min_size, max_size + 1))
        x = float(rng.integers(0, frame_width - w))
        y = float(rng.integers(0, frame_height - h))
        speed = float(rng.uniform(min_speed, max_speed))
        angle = float(rng.uniform(0, 2 * np.pi))
        dx = speed * np.cos(angle)
        dy = speed * np.sin(angle)
        color = tuple(int(c) for c in rng.integers(50, 256, size=3))
        shape = "circle" if random_shapes and rng.random() < 0.5 else "rect"
        objects.append(SyntheticObject(i, x, y, w, h, dx, dy, color, shape))
    return objects


def generate_occluders(
    num_occluders: int,
    frame_width: int,
    frame_height: int,
    min_size: int,
    max_size: int,
    min_speed: float,
    max_speed: float,
) -> List[Occluder]:
    """Generate a list of occluders with random properties."""
    occluders: List[Occluder] = []
    rng = np.random.default_rng()
    for _ in range(num_occluders):
        w = int(rng.integers(min_size, max_size + 1))
        h = int(rng.integers(min_size, max_size + 1))
        x = float(rng.integers(0, frame_width - w))
        y = float(rng.integers(0, frame_height - h))
        speed = float(rng.uniform(min_speed, max_speed))
        angle = float(rng.uniform(0, 2 * np.pi))
        dx = speed * np.cos(angle)
        dy = speed * np.sin(angle)
        # Colour the occluder dark grey rather than pure black so that
        # trackers still have some texture to latch onto.
        color = (30, 30, 30)
        occluders.append(Occluder(x, y, w, h, dx, dy, color))
    return occluders


def draw_scene(
    frame: np.ndarray,
    objects: List[SyntheticObject],
    occluders: Optional[List[Occluder]] = None,
    show_trajectory: bool = False,
) -> np.ndarray:
    
    for obj in objects:
        color = tuple(int(c) for c in obj.color)
        if obj.shape == "circle":
            center = (int(obj.x + obj.w / 2), int(obj.y + obj.h / 2))
            radius = int(min(obj.w, obj.h) / 2)
            cv2.circle(frame, center, radius, color, -1)
        else:
            x, y, w, h = obj.bounding_box()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        # Append current centre to trajectory for drawing later
        obj.trajectory.append((int(obj.x + obj.w / 2), int(obj.y + obj.h / 2)))

        # Limit trajectory length to avoid infinite growth
        max_traj_length = 50
        if len(obj.trajectory) > max_traj_length:
            obj.trajectory = obj.trajectory[-max_traj_length:]

    # Draw occluders on top
    if occluders is not None:
        for occ in occluders:
            x, y, w, h = occ.bounding_box()
            cv2.rectangle(frame, (x, y), (x + w, y + h), occ.color, -1)

    # Draw trajectories after occlusion so they remain visible
    if show_trajectory:
        for obj in objects:
            if len(obj.trajectory) > 1:
                # Use a faded colour for the trajectory
                traj_color = tuple(int(c * 0.5) for c in obj.color)
                cv2.polylines(
                    frame,
                    [np.array(obj.trajectory, dtype=np.int32)],
                    isClosed=False,
                    color=traj_color,
                    thickness=2,
                )
    return frame


def advanced_tracking(args: argparse.Namespace) -> None:
    """Run the advanced synthetic tracking demo based on provided arguments."""
    # Prepare synthetic objects
    objects = generate_synthetic_objects(
        num_objects=args.num_objects,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        min_size=args.min_size,
        max_size=args.max_size,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        random_shapes=args.random_shapes,
    )

    # Prepare occluders if requested
    occluders: Optional[List[Occluder]] = None
    if args.occlusion:
        occluders = generate_occluders(
            num_occluders=args.num_occluders,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            min_size=args.occluder_min_size,
            max_size=args.occluder_max_size,
            min_speed=args.occluder_min_speed,
            max_speed=args.occluder_max_speed,
        )

    # Set up trackers for each object
    trackers: Dict[int, cv2.Tracker] = {}
    initialised = False
    # Kalman filters for each object if requested
    kalman_filters: Dict[int, cv2.KalmanFilter] = {}
    # Data records for CSV
    records: List[Dict[str, float]] = []

    # Video writer if saving video
    writer: Optional[cv2.VideoWriter] = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.video_path,
            fourcc,
            args.fps,
            (args.frame_width, args.frame_height),
        )

    for frame_idx in range(args.num_frames):
        # Create blank frame
        frame = np.zeros(
            (args.frame_height, args.frame_width, 3), dtype=np.uint8
        )

        # Update positions of objects and occluders
        for obj in objects:
            obj.update_position(args.frame_width, args.frame_height)
        if occluders is not None:
            for occ in occluders:
                occ.update_position(args.frame_width, args.frame_height)

        # Draw objects and occluders onto the frame
        frame = draw_scene(
            frame,
            objects,
            occluders=occluders,
            show_trajectory=args.show_traj,
        )

        # Initialise trackers on the first frame
        if not initialised:
            for obj in objects:
                tracker = create_tracker(args.tracker_type)
                bbox = obj.bounding_box()
                tracker.init(frame, bbox)
                trackers[obj.id] = tracker
                # Create Kalman filter if enabled
                if args.use_kalman:
                    cx, cy = obj.center()
                    # Use measured velocities as initial state, although
                    # Kalman filter will quickly adapt
                    kf = create_kalman_filter((cx, cy, obj.dx, obj.dy))
                    kalman_filters[obj.id] = kf
            initialised = True
        else:
            # Update each tracker and optionally Kalman filter
            for obj in objects:
                tracker = trackers.get(obj.id)
                if tracker is None:
                    continue
                ok, bbox = tracker.update(frame)
                # Measured bounding box
                x, y, w, h = [int(v) for v in bbox]
                cx_meas = x + w / 2.0
                cy_meas = y + h / 2.0

                if args.use_kalman:
                    kf = kalman_filters.get(obj.id)
                    # Predict next state
                    pred_state = kf.predict()
                    pred_x, pred_y = pred_state[0, 0], pred_state[1, 0]
                    # Correct with measurement
                    measurement = np.array([[cx_meas], [cy_meas]], dtype=np.float32)
                    est_state = kf.correct(measurement)
                    est_x, est_y = est_state[0, 0], est_state[1, 0]
                    # Convert predicted centre back to top‑left
                    pred_top_left = (
                        int(est_x - w / 2.0),
                        int(est_y - h / 2.0),
                    )
                    # Draw predicted bounding box in blue
                    cv2.rectangle(
                        frame,
                        pred_top_left,
                        (pred_top_left[0] + w, pred_top_left[1] + h),
                        (255, 0, 0),
                        2,
                    )
                    # Draw a small circle at the predicted centre
                    cv2.circle(
                        frame,
                        (int(est_x), int(est_y)),
                        3,
                        (255, 0, 0),
                        -1,
                    )
                else:
                    # Without Kalman filter we treat measured centre as prediction
                    pred_x, pred_y = cx_meas, cy_meas

                # Draw measured bounding box in red
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2,
                )
                # Draw object ID
                cv2.putText(
                    frame,
                    f"ID {obj.id}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Record data for CSV
                record = {
                    "frame": frame_idx,
                    "id": obj.id,
                    "meas_x": x,
                    "meas_y": y,
                    "meas_w": w,
                    "meas_h": h,
                    "meas_cx": cx_meas,
                    "meas_cy": cy_meas,
                    "pred_cx": pred_x,
                    "pred_cy": pred_y,
                }
                records.append(record)

        # Write frame to video file if requested
        if writer is not None:
            writer.write(frame)

        # Display the frame unless running in headless mode.  If
        # no_display is True or a display cannot be opened, skip
        # showing the window. Catch exceptions thrown by imshow on
        # headless systems and continue gracefully.
        if not args.no_display:
            try:
                cv2.imshow("Advanced Synthetic Tracking", frame)
                # Exit if user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except Exception:
                # In case imshow fails (e.g., no display), set no_display
                # to True to avoid repeated exceptions in subsequent
                # iterations. We still continue to process frames and
                # write video/CSV if requested.
                args.no_display = True

    # Clean up
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Save CSV file if requested
    if args.save_csv and records:
        df = pd.DataFrame.from_records(records)
        df.to_csv(args.csv_path, index=False)
        print(f"Saved tracking data to {args.csv_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command‑line arguments for the advanced tracker."""
    parser = argparse.ArgumentParser(
        description="Advanced synthetic object tracking using OpenCV."
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=1,
        help="Number of synthetic objects to track.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
        help="Number of frames in the synthetic video.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Width of the video frame.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=480,
        help="Height of the video frame.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=30,
        help="Minimum width/height of synthetic objects.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=80,
        help="Maximum width/height of synthetic objects.",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=2.0,
        help="Minimum speed (pixels per frame) of objects.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=5.0,
        help="Maximum speed (pixels per frame) of objects.",
    )
    parser.add_argument(
        "--tracker-type",
        type=str,
        default="MIL",
        help="Tracker algorithm to use (currently only MIL is supported).",
    )
    parser.add_argument(
        "--use-kalman",
        action="store_true",
        help="Apply a Kalman filter to smooth object centres.",
    )
    parser.add_argument(
        "--show-traj",
        action="store_true",
        help="Draw the trajectory of each object on the frame.",
    )
    parser.add_argument(
        "--random-shapes",
        action="store_true",
        help="Randomly choose between rectangles and circles for objects.",
    )
    parser.add_argument(
        "--occlusion",
        action="store_true",
        help="Enable moving occluders to simulate partial occlusion.",
    )
    parser.add_argument(
        "--num-occluders",
        type=int,
        default=2,
        help="Number of occluders in the scene (only relevant if occlusion is enabled).",
    )
    parser.add_argument(
        "--occluder-min-size",
        type=int,
        default=50,
        help="Minimum size of occluders.",
    )
    parser.add_argument(
        "--occluder-max-size",
        type=int,
        default=120,
        help="Maximum size of occluders.",
    )
    parser.add_argument(
        "--occluder-min-speed",
        type=float,
        default=1.0,
        help="Minimum speed of occluders.",
    )
    parser.add_argument(
        "--occluder-max-speed",
        type=float,
        default=3.0,
        help="Maximum speed of occluders.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the resulting video to disk.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="synthetic_tracking.mp4",
        help="Path to save the output video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the saved video.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save tracking data to a CSV file.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="tracking_data.csv",
        help="Path to save the CSV data.",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Run in headless mode without displaying GUI windows. This is "
            "useful when executing on servers or environments without a "
            "graphics display."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    # If saving CSV, ensure directory exists
    if args.save_csv:
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
    if args.save_video:
        os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
    advanced_tracking(args)


if __name__ == "__main__":
    main()