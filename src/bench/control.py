from src.core.errors import ConfigError, ControlViolationError

REQUIRED_EXCEPTION_KEYS = ("split", "field", "value", "reason", "approved_by", "approved_at")


def validate_exceptions(exceptions):
    for exception in exceptions:
        missing = [key for key in REQUIRED_EXCEPTION_KEYS if key not in exception]
        if missing:
            raise ConfigError(
                f"control.exceptions entry is missing required keys {missing}: {exception}. "
                f"PLAN-P1 SS10.2 requires split/field/value/reason/approved_by/approved_at."
            )

CONTROL_FIELDS = [
    "runtime.seed",
    "runtime.amp",
    "runtime.deterministic",
    "runtime.device",
    "runtime.allow_network",
    "data.name",
    "data.root",
    "data.params",
    "data.split",
    "data.image_size",
    "data.transform",
    "data.num_workers",
    "data.drop_last",
    "data.batch_size",
    "loss",
    "metrics",
    "adapter",
    "optim.optimizer",
    "optim.scheduler",
    "train.epochs",
    "train.monitor",
    "train.grad_clip",
]


def get_by_path(config, dotted_path):
    node = config
    for part in dotted_path.split("."):
        node = node[part]
    return node


def build_control_report(split_configs, exceptions, extra_fields=None):
    """extra_fields lets a benchmark config declare additional dotted paths to control on top of
    CONTROL_FIELDS (bench yaml `control.extra_fields`). This is a generic, config-driven
    extension point -- any task can use it, none is named here -- for promoting task-specific
    postprocessing hyperparameters to controlled fields without hardcoding a task name into
    this module (Grade B contract extension, PLAN-P1 SS16)."""
    validate_exceptions(exceptions)
    exceptions_index = {(e["split"], e["field"]): e for e in exceptions}
    split_names = list(split_configs.keys())
    reference_split = split_names[0]

    fields_report = {}
    violations = []
    applied_exceptions = []

    for field in list(CONTROL_FIELDS) + list(extra_fields or []):
        reference_value = get_by_path(split_configs[reference_split], field)
        field_report = {reference_split: reference_value}
        for split_name in split_names[1:]:
            value = get_by_path(split_configs[split_name], field)
            field_report[split_name] = value
            if value == reference_value:
                continue
            key = (split_name, field)
            exception = exceptions_index.get(key)
            if exception is not None and exception["value"] == value:
                applied_exceptions.append(exception)
                continue
            violations.append({
                "split": split_name, "field": field,
                "reference": reference_value, "value": value,
            })
        fields_report[field] = field_report

    return {
        "reference_split": reference_split,
        "fields": fields_report,
        "violations": violations,
        "exceptions_applied": applied_exceptions,
        "extra_fields": list(extra_fields or []),
    }


def enforce_control(split_configs, exceptions, extra_fields=None):
    report = build_control_report(split_configs, exceptions, extra_fields)
    if report["violations"]:
        raise ControlViolationError(
            f"Unapproved control field differences across splits: {report['violations']}"
        )
    return report
