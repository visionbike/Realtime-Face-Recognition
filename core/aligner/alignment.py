import cv2
import numpy as np
from skimage import transform as sktform


# Define a standard set of destination landmarks for ArcFace alignment
ARCFACE_STD_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def get_alignment_matrix(landmarks: np.ndarray, image_size: int = 112, mode: str = "arcface") -> np.ndarray:
    """
    Estimate the transformation matrix for aligning facial landmarks.

    :param landmarks: 2D array of shape (5,2) representing the facial landmarks.
    :param image_size: Desired output image size.
    :param mode: Alignment mode, currently only "arcface" is supported.
    :return: Transformation matrix (2x3) for aligning facial landmarks.
    """
    # check input conditions
    assert landmarks.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0

    if mode == "arcface":
        # adjust ratio and x-coordinate difference based on image size
        if image_size % 112 == 0:
            ratio = float(image_size) / 112
            offset_x = 0
        else:
            ratio = float(image_size) / 128
            offset_x = 8.0 * ratio

        # scale and shift the destination landmarks
        dest_landmarks = ARCFACE_STD_LANDMARKS * ratio
        dest_landmarks[:, 0] += offset_x
    else:
        raise ValueError(f"Unsupported alignment mode: {mode}.")

    # estimate the similarity transformation
    similarity_tform = sktform.SimilarityTransform.from_estimate(landmarks, dest_landmarks)
    if not similarity_tform:
        raise ValueError("Failed to estimate alignment transform from landmarks.")
    return similarity_tform.params[0: 2, :]


def align_face(image: np.ndarray, landmark: np.ndarray, image_size: int = 112, mode: str = "arcface") -> np.ndarray:
    """
    Normalize and crop a facial image based on provided landmarks.

    :param image: Input facial image.
    :param landmark: 2D array of shape (5, 2) representing the facial landmarks.
    :param image_size: Desired output image size.
    :param mode: Alignment mode, currently only "arcface" is supported.
    :return: Normalized and cropped facial image.
    """
    # estimate the transformation matrix
    alignment_matrix = get_alignment_matrix(landmark, image_size, mode)

    # apply the affine transformation to the image
    warped_image = cv2.warpAffine(image, alignment_matrix, (image_size, image_size), borderValue=0.0)
    return warped_image
