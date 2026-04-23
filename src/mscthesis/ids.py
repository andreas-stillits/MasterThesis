from __future__ import annotations

from .log import log_call


@log_call()
def validate_sample_id(sample_id: str, required_digits: int) -> str:
    """
    Validate that the sample ID matches the required lenght and is mappable to int
    Args:
        sample_id (str): The sample ID to validate.
        required_digits (int): The required length of the sample ID.
    """
    # if not required number of digits
    if not len(sample_id) == required_digits:
        raise ValueError(
            f"Sample ID '{sample_id}' does not match required "
            f"length of {required_digits} digits."
        )
    # if not mappable to int
    try:
        int(sample_id)
    except Exception as exc:
        raise ValueError(
            f"Sample ID '{sample_id}' is not a valid integer string."
        ) from exc

    return sample_id
