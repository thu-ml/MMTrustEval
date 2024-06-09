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
│   │   ├── mock.py
│   │   ├── unrelatedimg.py
│   │   ├── vispr.py
│   │   └── vizwiz.py
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
│   │   └── unrelated_noise.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── base.py
│   ├── tasks
│   │   └── base.py
│   └── utils
│       ├── __init__.py
│       ├── registry.py
│       └── utils.py
├── run_task.py

  
```


## Task Flow

<figure markdown="span">
  ![Image title](structure/image-20240522145344912.png){ width="300" }
  <figcaption>Task Flow</figcaption>
</figure>
