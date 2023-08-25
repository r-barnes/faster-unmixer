from typing import Optional

import cvxpy as cp


class ReciprocalParameter:
    """
    Used for times when you want a cvxpy Parameter and its ratio.

    Attributes:
        p (cp.Parameter): The original parameter.
        rp (cp.Parameter): The reciprocal of the original parameter.
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the ReciprocalParameter.

        Note:
            The ReciprocalParameter creates two underlying cp.Parameters: `p` and `rp`, which represent
            the original parameter and its reciprocal, respectively.
        """

        self._p = cp.Parameter(*args, **kwargs)
        # Reciprocal of the above
        self._rp = cp.Parameter(*args, **kwargs)

    @property
    def value(self) -> Optional[float]:
        """
        Get the value of the ReciprocalParameter.

        Returns:
            Optional[float]: The value of the original parameter.
        """
        return self._p.value

    @value.setter
    def value(self, val: Optional[float]) -> None:
        """
        Set the value of the ReciprocalParameter and its reciprocal.

        Args:
            val (Optional[float]): The value to be set for the original parameter.

        Note:
            The method sets the value of the original parameter (`_p`) to the specified value (`val`), and
            sets the value of the reciprocal parameter (`_rp`) to 1/val.
        """
        self._p.value = val
        self._rp.value = 1 / val if val is not None else None

    @property
    def p(self) -> cp.Parameter:
        """
        Get the original parameter.

        Returns:
            cp.Parameter: The original parameter.
        """
        return self._p

    @property
    def rp(self) -> cp.Parameter:
        """
        Get the reciprocal of the parameter.

        Returns:
            cp.Parameter: The reciprocal of the parameter.
        """
        return self._rp


def cp_log_ratio(a: cp.Variable, b: ReciprocalParameter) -> cp.Expression:
    """
    Returns a convex version of the log-ratio of a CVXPY variable and a Parameter.

    Args:
        a (cp.Variable): The CVXPY variable.
        b (ReciprocalParameter): The ReciprocalParameter representing the parameter value.

    Returns:
        cp.Expression: A convex expression representing a substitute for log-ratio of a and b.
    """
    return cp.maximum(a * b.rp, b.p * cp.inv_pos(a))
