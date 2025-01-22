import logging

log = logging.getLogger(__name__)

NUM_CSTS_ID = 8

JET_FEATURES = [
    "jets",
    "labels",
]

CST_FEATURES = [
    "csts",
    "csts_id",
    "vtx_id",
    "vtx_pos",
    "mask",
    "track_type",
]

JC_CLASS_TO_LABEL = {
    "ZJetsToNuNu": 0,
    "TTBarLep": 1,
    "TTBar": 2,
    "WToQQ": 3,
    "ZToQQ": 4,
    "HToBB": 5,
    "HToCC": 6,
    "HToGG": 7,
    "HToWW4Q": 8,
    "HToWW2Q1L": 9,
}

BT_CLASS_TO_LABEL = {
    "light": 0,
    "charm": 1,
    "bottom": 2,
}
