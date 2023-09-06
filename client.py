import argparse

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, Swinv2ForImageClassification

ACTIONS = [
    "Normal",
    "Drink",
    "Look Left",
    "Look Right",
    "Operate Dashboard",
    "Take Object",
    "Talk Cell",
    "Texting",
]

ACTION_THRESHOLD = [0.15, 0.4, 0.15, 0.15, 0.3, 0.3, 0.5, 0.5]
LOGITS_MULT = [1.0, 0.9, 1.0, 1.0, 1.1, 1.0, 0.9, 1.0]
EXP_DECAY = 0.95
LOGITS_THRESHOLD = 10.0
DISTRIB_MAX = LOGITS_THRESHOLD / (1.0 - EXP_DECAY)
SELECT_THRESHOLD = np.array(ACTION_THRESHOLD) * DISTRIB_MAX

model_path = "final_model_without_maria_91percento.pt"

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = AutoImageProcessor.from_pretrained(
    "microsoft/swinv2-small-patch4-window16-256"
)

model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-small-patch4-window16-256"
)


class Hysteresis:
    def __init__(self):
        self.distrib = None

    def process(self, predicted):
        predicted = predicted * LOGITS_MULT
        predicted[predicted < LOGITS_THRESHOLD] = 0
        imax = np.argmax(predicted)
        if self.distrib is None:
            self.distrib = np.zeros_like(predicted)
        self.distrib *= EXP_DECAY
        self.distrib += predicted
        return self.distrib

    def predicted_class(self):
        imax = np.argmax(self.distrib)
        return imax if (self.distrib[imax] > SELECT_THRESHOLD[imax]) else -1


HLCLR = (250, 140, 90)
HL2CLR = (90, 190, 250)


def alphaCompose(frgnd, bkgnd):
    frgnd = frgnd.astype(float)
    bkgnd = bkgnd.astype(float)
    out = cv2.add(frgnd, bkgnd)
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def logit2clr(l):
    if l >= LOGITS_THRESHOLD:
        return HLCLR
    x = 1.0 - min(1.0, abs(l - LOGITS_THRESHOLD) / (LOGITS_THRESHOLD))
    x2 = int(80 + (200 - 80) * x)
    return (x2, x2, x2)


def d2clr(d, dmin):
    x = min(d, dmin) / dmin
    x2 = int(80 + (200 - 80) * x)
    return (x2, x2, x2)


OUT_VIDEO_SZ = (400, 530)


def prepare_video_image(frame, logits, distrib, stab_class):
    frame2 = cv2.resize(frame, (400, 300))
    frame2 = np.vstack(
        [
            np.zeros((50, 400, 3), dtype=np.uint8),
            frame2,
            np.zeros((180, 400, 3), dtype=np.uint8),
        ]
    )

    for ii in range(len(ACTIONS)):
        color = logit2clr(logits[ii])
        d = min(1.0, distrib[ii] / DISTRIB_MAX)
        if d < 0.001:
            d = 0.0
        dcolor = HL2CLR if ii == stab_class else d2clr(d, ACTION_THRESHOLD[ii])
        cv2.putText(
            frame2,
            f"{ACTIONS[ii]}",
            (10, 355 + 20 * (ii + 1)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame2,
            f"{logits[ii]:.4}",
            (200, 355 + 20 * (ii + 1)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame2,
            f"{d:.4}",
            (300, 355 + 20 * (ii + 1)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            dcolor,
            2,
            cv2.LINE_AA,
        )
    if stab_class >= 0:
        cv2.putText(
            frame2,
            f"{ACTIONS[stab_class]}",
            (10, 35),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            HL2CLR,
            2,
            cv2.LINE_AA,
        )
    return frame2


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", required=True, type=str, help="Model path")
    argparser.add_argument(
        "--display", action="store_true", default=False, help="Interactive mode"
    )
    argparser.add_argument(
        "--calibrate", action="store_true", default=False, help="Show calibration mask"
    )
    args = argparser.parse_args()

    model.classifier = torch.nn.Linear(768, 8, bias=True)
    model.load_state_dict(torch.load(args.model, map_location=device))

    if args.calibrate:
        VIDEO_MASK = cv2.imread("calibration_mask0.png", cv2.IMREAD_COLOR)
    camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 25)

    print("Starting loop")
    h = Hysteresis()
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        with torch.inference_mode():
            x = feature_extractor(images=frame, return_tensors="pt")["pixel_values"]
            logits = model(x.to(device)).logits.numpy()

        distrib = h.process(logits[0])
        stab_class = h.predicted_class()

        if args.display:
            if args.calibrate:
                frame = alphaCompose(frame, VIDEO_MASK)

            frame2 = prepare_video_image(frame, logits[0], distrib, stab_class)
            cv2.imshow("Driver Activity Recognition", frame2)
            if cv2.waitKey(1) & 0xFF != 255:
                break

    print("\nLoop stopped")
    camera.release()
    cv2.destroyAllWindows()
    # client.disconnect()


if __name__ == "__main__":
    main()
