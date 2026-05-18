# Model Hosting with Google Cloud Storage

This guide explains how to manage Barrikada models using Google Cloud Storage (GCS) for centralized storage and distribution.

## Overview

Models are organized in a centralized `core/models/` directory and uploaded to a GCS bucket for team access:

- **Local Development**: Models downloaded from GCS to `core/models/` and cached locally
- **Production**: Models downloaded at container startup from GCS
- **Versioning**: Current models in main path, older versions archived with timestamps
- **No Runtime Changes**: Runtime loading code unchanged—always reads from local `core/models/`

```
core/models/
├── layer_b/
│   ├── *.faiss (FAISS indices)
│   ├── *.npy (embeddings)
│   ├── *.json (metadata)
│   └── archives/
│       └── backup_YYYYMMDD_HHMMSS/
├── layer_c/
│   ├── *.joblib (classifiers)
│   └── archives/
├── layer_d/
│   ├── model/ (Hugging Face weights)
│   ├── config.json
│   └── archives/
└── layer_e/
    ├── qwen3guard-barrikade/ (Qwen3Guard-Gen-0.6B bundle)
    └── archives/
```

## Setup

### 1. GCS Bucket Setup

**Create a GCS bucket** (if not already done):

```bash
export PROJECT_ID=my-gcp-project
export BUCKET_NAME=barrikade-models

gsutil mb gs://$BUCKET_NAME
gsutil versioning set on gs://$BUCKET_NAME  # optional: enable versioning
gsutil logging set on -b gs://barrikade-logs -o model-uploads gs://$BUCKET_NAME  # optional: logs
```

### 2. Local Development Setup

**Install dependencies**:

```bash
pip install -r requirements.txt
```

**Download Layer E from Hugging Face**:

```bash
python scripts/download_qwen3guard.py
```

**Initialize models directory** (one-time):

```bash
# Create directory structure
mkdir -p core/models/{layer_b,layer_c,layer_d,layer_e}/archives

# If you already have local models in layer_*/outputs/, bundle them:
python scripts/bundling/bundle_models.py --manifest core/models/manifest.json

# Validate bundled models:
python scripts/bundling/validate_models.py --verbose
```

### 3. Upload Models to GCS (First Time)

**Prerequisites for uploading**:
- Service account credentials with write access (created above)
- Credentials set: `export GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/barrikade-credentials.json`

**Bundle any new models** from layer outputs:

```bash
python scripts/bundling/bundle_models.py --archive-old --manifest core/models/manifest.json
```

**Upload to GCS**:

```bash
python scripts/bundling/gcs_upload.py \
    --bucket barrikade-bundles 
```

This uploads all files from `core/models/` to:
```
gs://barrikade-bundles/models/layer_b/
gs://barrikade-bundles/models/layer_c/
gs://barrikade-bundles/models/layer_d/
gs://barrikade-bundles/models/layer_e/
```

and archives any previous version to:
```
gs://barrikade-bundles/models/layer_b/archives/backup_YYYYMMDD_HHMMSS/
```

## Workflows

### Workflow 1: Team Member Downloads Latest Models

**Downloads use public bucket access (no credentials needed):**

```bash
# Download latest models from GCS to local core/models/
python scripts/bundling/gcs_download.py \
    --bucket barrikade-bundles \
    --archive-old \
    --validate

# Now the pipeline uses local models
python -c "from core.orchestrator import PIPipeline; p = PIPipeline()"
```

### Workflow 2: Developer Trains New Model and Uploads

```bash
# Train model (e.g., layer_c classifier)
# Output goes to core/layer_c/outputs/

# Bundle the trained model into core/models/
python scripts/bundling/bundle_models.py --archive-old

# Upload to GCS
python scripts/bundling/gcs_upload.py --bucket barrikade-models

# Other team members can now download:
python scripts/bundling/gcs_download.py --bucket barrikade-models
```

### Workflow 3: Revert to Previous Model Version

Models are automatically archived when new versions are uploaded. To use an older version:

```bash
# List available archived versions locally
python scripts/bundling/model_status.py --bucket barrikade-models

# Download specific archived version (from GCS archives)
python scripts/bundling/gcs_download.py \
    --bucket barrikade-models \
    --version archives \
    --layers layer_c

# This downloads from gs://barrikade-models/models/layer_c/archives/
```

### Workflow 4: Monitor Model Status

Compare local vs. GCS models:

```bash
python scripts/bundling/model_status.py --bucket barrikade-models

# Output:
# Layer B: 5 files locally, 5 files in GCS, in sync ✓
# Layer C: 8 files locally, 8 files in GCS, in sync ✓
# Layer D: 3 dirs locally, 3 dirs in GCS, in sync ✓
# Layer E: 2 dirs locally, 2 dirs in GCS, in sync ✓
```

### Workflow 5: Clean Up Old Archives

Keep storage costs down by removing old archived versions:

```bash
# Clean local archives (keep 3 most recent versions)
python scripts/bundling/cleanup_archives.py --keep 3 --local

# Clean GCS archives (keep 3 most recent versions)
python scripts/bundling/cleanup_archives.py \
    --keep 3 \
    --gcs \
    --bucket barrikade-models \
    --dry-run  # Preview what would be deleted

# Actually delete
python scripts/bundling/cleanup_archives.py \
    --keep 3 \
    --gcs \
    --bucket barrikade-models
```

## Docker Deployment

### Configuration

Optionally set the public GCS bucket name override (no credentials needed):

```yaml
services:
    barrikade-api:
        environment:
            BARRIKADA_GCS_BUCKET: barrikade-models
```

### Startup Process

When the container starts:

1. **Check for local models** at `/app/core/models/`
2. If valid local models are present:
     - Use them directly
3. If not found or invalid:
     - Download models from public GCS bucket using anonymous access
     - Validate downloaded models
4. Start the API server

### Example: Start Container with GCS Models

```bash
docker compose build

docker compose run barrikade-api
```

### Local Development with Docker

Mount local models to skip GCS download:

```bash
docker compose run -v $(pwd)/core/models:/app/core/models:ro barrikade-api
```

Or override the public GCS bucket explicitly:

```bash
docker compose run -e BARRIKADA_GCS_BUCKET=barrikade-models barrikade-api
```

## Command Reference

### `bundle_models.py`

Bundle scattered layer outputs into centralized `core/models/` directory.

```bash
python scripts/bundling/bundle_models.py \
    [--dry-run] \
    [--archive-old] \
    [--manifest PATH]
```

**Options**:
- `--dry-run`: Show what would be bundled without making changes
- `--archive-old`: Move current models to archives before bundling new ones
- `--manifest PATH`: Save bundle manifest to JSON file

**Output**: Consolidates models from `core/layer_*/outputs/` to `core/models/`

---

### `validate_models.py`

Validate that bundled models are present and loadable.

```bash
python scripts/bundling/validate_models.py [--verbose]
```

**Checks**:
- All required files present for each layer
- Models can be loaded without errors
- Archive directories exist

---

### `gcs_upload.py`

Upload bundled models to GCS bucket.

```bash
python scripts/bundling/gcs_upload.py \
    --bucket BUCKET_NAME \
    [--project PROJECT_ID] \
    [--layers layer_b,layer_c] \
    [--no-archive] \
    [--dry-run] \
    [--manifest PATH]
```

**Options**:
- `--bucket` (required): GCS bucket name
- `--project`: GCP project ID (auto-detected if not specified)
- `--layers`: Comma-separated layer names (default: all)
- `--archive`: Archive previous version before uploading (default: true)
- `--dry-run`: Preview upload without making changes
- `--manifest`: Save upload manifest to JSON

**Requires**: `GOOGLE_APPLICATION_CREDENTIALS` environment variable (write access to bucket)

---

### `gcs_download.py`

Download models from PUBLIC GCS bucket to local `core/models/`.

```bash
python scripts/bundling/gcs_download.py \
    --bucket BUCKET_NAME \
    [--layers layer_b,layer_c] \
    [--version latest|archives] \
    [--no-archive-old] \
    [--no-validate] \
    [--manifest PATH]
```

**Options**:
- `--bucket` (required): GCS bucket name (must be publicly readable)
- `--layers`: Comma-separated layer names (default: all)
- `--version`: `latest` for current models, `archives` for older versions (default: latest)
- `--archive-old`: Backup current models before downloading (default: true)
- `--validate`: Validate after download (default: true)
- `--manifest`: Save download manifest to JSON

**Note**: Uses anonymous access - no credentials required.

---

### `model_status.py`

Show status of local models vs. GCS models.

```bash
python scripts/bundling/model_status.py \
    --bucket BUCKET_NAME \
    [--layers layer_b,layer_c]
```

**Options**:
- `--bucket` (required): GCS bucket name
- `--layers`: Comma-separated layer names (default: all)

**Output**: Local/GCS file counts, sizes, timestamps, and sync status

**Credentials**:
- **Public buckets**: No credentials needed
- **Private buckets**: Requires `GOOGLE_APPLICATION_CREDENTIALS` environment variable

---

### `cleanup_archives.py`

Remove old archived model versions.

```bash
python scripts/bundling/cleanup_archives.py \
    [--keep N] \
    [--layers layer_b,layer_c] \
    [--local] \
    [--gcs] \
    [--bucket BUCKET_NAME] \
    [--dry-run]
```

**Options**:
- `--keep`: Number of recent versions to keep (default: 3)
- `--layers`: Comma-separated layer names (default: all)
- `--local`: Clean local archives (default: true if --gcs not specified)
- `--gcs`: Clean GCS archives (requires --bucket and write permissions)
- `--bucket`: GCS bucket name (required if --gcs specified)
- `--dry-run`: Preview deletions without making changes

**Credentials**:
- **Local cleanup**: No credentials needed
- **GCS cleanup**: Requires `GOOGLE_APPLICATION_CREDENTIALS` with write access

---

## Troubleshooting

### "Models not found" error in container

**Cause**: Container cannot access public GCS bucket or models are not in `core/models/`

**Fix**:
1. Verify bucket is publicly readable: `gsutil iam get gs://$BARRIKADA_GCS_BUCKET | grep allUsers`
2. Check bucket contents: `gsutil ls gs://$BARRIKADA_GCS_BUCKET/models/`
3. Check container logs: `docker-compose logs barrikade-api`
4. Test download manually: `python scripts/bundling/gcs_download.py --bucket barrikade-models`
5. Mount models locally: `docker-compose run -v $(pwd)/core/models:/app/core/models`

### "Access Denied" error

**Cause**: Bucket is not publicly readable

**Fix**:
```bash
# Make bucket publicly readable
gsutil iam ch allUsers:objectViewer gs://$BUCKET_NAME

# Verify it worked
gsutil iam get gs://$BUCKET_NAME | grep allUsers
```

### Models fail validation after download

**Cause**: Incomplete or corrupted download

**Fix**:
```bash
# Delete and re-download
rm -rf core/models/layer_*/*.joblib core/models/layer_*/model core/models/layer_e/qwen3guard-barrikade core/models/layer_*/teacher

# Re-download with validation
python scripts/bundling/gcs_download.py --bucket barrikade-models --validate

# Check validation
python scripts/bundling/validate_models.py --verbose
```

### Pipeline still using old model versions

**Cause**: Local cache not invalidated

**Fix**:
```bash
# Force re-download by backing up existing models
python scripts/bundling/gcs_download.py --bucket barrikade-models --archive-old

# Or delete and download fresh
rm -rf core/models
python scripts/bundling/gcs_download.py --bucket barrikade-models
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BARRIKADA_GCS_BUCKET` | - | GCS bucket name for model storage |
| `BARRIKADA_GCS_PROJECT` | - | GCP project ID (auto-detected if not set) |
| `GOOGLE_APPLICATION_CREDENTIALS` | - | Path to GCP service account JSON file |
| `BARRIKADA_ARTIFACTS_DIR` | `./artifacts` | Local artifact storage directory |

## Cost Optimization

### Reduce Storage Costs

1. **Archive cleanup**: Keep only recent versions
   ```bash
   python scripts/bundling/cleanup_archives.py --keep 2 --gcs --bucket barrikade-models
   ```

2. **Object lifecycle**: Configure GCS lifecycle policy to delete old backups
   ```json
   {
     "lifecycle": {
       "rule": [{
         "action": {"type": "Delete"},
         "condition": {"age": 90, "matchesPrefix": ["models/*/archives/"]}
       }]
     }
   }
   ```

3. **Compression**: Models can be compressed before upload (not implemented yet)

### Reduce Egress Costs

1. **Mirror bucket in regions** where team is located
2. **Use GCS caching** in containers
3. **Batch downloads** - download all needed models at once

## Security Considerations

1. **Service Account Permissions**: Restrict to storage.objectAdmin on specific bucket
   ```bash
   gcloud projects add-iam-policy-binding $PROJECT_ID \
       --member=serviceAccount:barrikade-models@$PROJECT_ID.iam.gserviceaccount.com \
       --role=roles/storage.objectViewer  # Read-only if downloads only
   ```

2. **Credential Management**: 
   - Never commit credentials to Git
   - Rotate keys periodically
   - Use Workload Identity in GKE instead of keys

3. **Bucket Privacy**: 
   - Enable uniform bucket-level access
   - Block public access
   - Enable audit logging

```bash
gsutil uniformbucketlevelaccess set on gs://$BUCKET_NAME
gsutil logging set on -b gs://barrikade-logs gs://$BUCKET_NAME
```

## FAQ

**Q: Can I use models without GCS?**  
A: Yes. Place models directly in `core/models/` directory. Container startup will use local models if GCS download fails or credentials are not available.

**Q: How often should I bundle and upload?**  
A: After training a new model or when distributing new versions to the team. Can be automated in CI/CD.

**Q: Do I need to update settings.py?**  
A: No. Runtime loading code unchanged. Models are always read from `core/models/` locally.

**Q: How much storage do models use?**  
A: Check with `python scripts/bundling/model_status.py --bucket barrikade-models`

**Q: Can multiple team members upload simultaneously?**  
A: Yes, but only one upload per layer at a time to avoid conflicts. GCS handles concurrent reads safely.

**Q: What if internet is down?**  
A: Container will fail to start if models aren't cached locally. Pre-download models or use local mounts for development.
