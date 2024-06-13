# Structure


## Project Structure

```
├── mmte
│   ├── __init__.py
│   ├── configs
│   │   ├── __init__.py
│   │   ├── datasets/*
│   │   ├── models/*
│   │   └── task/*
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── celeb.py
│   │   ├── confaide.py
│   │   ├── enron_email.py
│   │   ├── vispr.py
│   │   └── ...
│   ├── evaluators
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chatmodel_eval.py
│   │   ├── classifier_eval.py
│   │   ├── metrics.py
│   │   └── rule_eval.py
│   ├── methods
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── related.py
│   │   ├── unrelated_color.py
│   │   ├── unrelated_nature.py
│   │   ├── unrelated_noise.py
│   │   └── ...
│   ├── models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai_chat.py
│   │   └──...
│   ├── tasks
│   │   └── base.py
│   └── utils
│       ├── __init__.py
│       ├── registry.py
│       └── utils.py
└── run_task.py

  
```


## <a name="flow"></a> Task Workflow

<figure markdown="span">
  ![Image title](structure/image-20240522145344912.png){ width="300" }
  <figcaption>Task Workflow</figcaption>
</figure>

The basic workflow of a task in MMTrustEval follows the pipeline above. The image-text pairs (or text-only samples) are retrieved from the customized `dataset`. They are likely to be further processed with a pre-defined method (e.g., pairing text with synthesized images, imposing adversarial noises to the images) by `method_hook` passed into the dataset. Data in multiple modalities is gathered into a dataclass, `TxtSample` or `ImageTxtSample`.  The samples ready for inference are then input to MLLMs with unified interface for `chat`. Further, the generated content is processed by diverse `evaluators` (e.g., keyword extraction, GPT-4 rating, classifier) and further standardized to be computed with specified `metrics` (e.g., accuracy, pearson correlation coefficient).
