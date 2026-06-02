# Experiment Registry (ICML/NeurIPS upgrade)

Every experiment run on the `icml-neurips-upgrade` branch is logged here
(appended automatically by `scripts_icml/icml_common.py::log_experiment`).

Fields logged per experiment: dataset, category/domain, seed, split ID, model,
features used, whether `product_average_rating` is included, whether
disagreement labels are used during training, key metrics, the command run, and
output file paths.

| timestamp | phase | dataset | category | seed | split_id | model | features | includes_product_avg_rating | disagreement_labels_in_training | command | output_files | key_metrics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
