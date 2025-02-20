from typing import Dict, List, Union

SingleMetricValue = Union[float, int, str, None]
MetricValueType = Union[SingleMetricValue, List[SingleMetricValue], Dict[str, SingleMetricValue]]
