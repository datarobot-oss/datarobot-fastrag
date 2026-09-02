# FastRAG Rollout — Deployment Flowchart

```mermaid
flowchart TD
    Start([Customer wants to\ndeploy a custom model])

    Start --> CreateVersion["Create custom model version\nin DataRobot UI"]

    CreateVersion --> PhaseCheck{Which milestone\nphase are we in?}

    PhaseCheck -->|"M1–M3 · POC\nSeparate images"| POCChoice["Customer explicitly selects\nan execution environment:\n  FastRAG staging image  —or—\n  Legacy DRUM-only image"]

    PhaseCheck -->|"M4–M6 · Merged image\nflag-controlled"| FlagCheck{"Flag GENAI_RAG_FRAG_RUNNER\nenabled for this org?"}

    PhaseCheck -->|"M7+ · DRUM deprecated"| FDOnly["Only FastRAG image available\nCustomer has no DRUM option"]

    FlagCheck -->|"Disabled · default off\nM4–M5 initial state"| DefaultDRUM["Default: DRUM\nCustomer can override to\nmerged image manually"]

    FlagCheck -->|"Enabled per-org\nM5 PRIVATE_PREVIEW opt-in"| DefaultFD["Default: Merged FastRAG image\nCustomer can override back\nto DRUM if needed"]

    FlagCheck -->|"Enabled by default\nM6 GA_PREMIUM"| DefaultFD

    POCChoice & DefaultDRUM & DefaultFD & FDOnly --> Freeze

    Freeze["⚙️  Platform resolves and FREEZES\nexecution environment image digest\ninto model version record\n\nbase_environment_version_id = sha256:…\n\nThis binding is permanent.\nNo flag change, platform update,\nor redeployment can alter it."]

    Freeze --> ModelVersion[(Model version created\nwith frozen image digest)]

    ModelVersion --> ExistingCheck{New or existing\ndeployment?}

    ExistingCheck -->|New deployment| Deploy["Customer deploys\nmodel version"]
    ExistingCheck -->|"Existing deployment\n— flag changed"| Frozen["❌  No effect on existing deployment\nFrozen image is permanent\nDRUM keeps running unchanged"]

    Deploy --> Container["Container starts\nstart_server.sh runs"]

    Container --> ImageType{Image type?}

    ImageType -->|"DRUM-only image\n(M1–M3 or pre-flag)"| DrumServer["drum server\nFlask · WSGI · sync"]

    ImageType -->|"Merged image\n(M4+)"| EnvVarCheck{"DR_GENAI_RAG_FRAG_RUNNER\nenv var injected\nby platform?"}

    EnvVarCheck -->|"Yes — flag ON\nfor this org"| FDServer["fastrag server\nFastAPI · ASGI · async"]

    EnvVarCheck -->|"No — flag OFF\nfor this org"| DrumServer

    DrumServer --> ServeDRUM([Serving predictions\nvia DRUM])
    FDServer --> ServeFD([Serving predictions\nvia FastRAG ✅])

    WantSwitch(["Customer on existing DRUM deployment\nwants to switch to FastRAG"]) --> CanToggle{"Can I toggle the flag\non the existing deployment?"}

    CanToggle -->|"No — image is\npermanently frozen"| MustCreate["Customer must create a\nNEW custom model version\npointing at the merged image"]

    MustCreate --> CreateVersion

    style Freeze fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style ModelVersion fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style ServeFD fill:#d1fae5,stroke:#10b981,color:#064e3b
    style ServeDRUM fill:#fef3c7,stroke:#f59e0b,color:#78350f
    style Frozen fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    style CanToggle fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    style MustCreate fill:#ffedd5,stroke:#f97316,color:#7c2d12
```

## Reading the diagram

| Colour | Meaning |
|--------|---------|
| 🔵 Blue | Platform-side immutable operation (freeze) |
| 🟢 Green | FastRAG serving |
| 🟡 Yellow | Legacy DRUM serving |
| 🔴 Red | Dead end — this path doesn't work |

## Key insight

The feature flag `GENAI_RAG_FRAG_RUNNER` controls **which environment is the default when creating a new model version**.
It does **not** affect existing deployments. Because the image digest is frozen at model-version creation time,
a model built against the old DRUM image will always run DRUM — forever — regardless of any flag change.

"Opt-in" = creating a new model version. Not flipping a switch.
