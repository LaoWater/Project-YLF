# --- Define your variables (copy and paste each line) ---
$PROJECT_ID = "terapie-acasa"
$REGION = "europe-central2"
$AR_REPO = "my-docker-repo" # <<< CHOOSE OR REPLACE YOUR AR REPO NAME
$IMAGE_NAME = "langgraph-api"
$IMAGE_TAG = $(Get-Date -Format yyyyMMdd-HHmmss)
$AR_IMAGE_PATH = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
$SERVICE_NAME = "langgraph-api" # <<< REPLACE WITH YOUR CLOUD RUN SERVICE NAME IF DIFFERENT
$BUILD_SA = "39620233431-compute@developer.gserviceaccount.com" # <<< CONFIRM THIS SA FROM YOUR ERROR MESSAGE

# --- One-time Setup Steps (copy and paste each line if not done) * ONly one time---
# 1. Create Artifact Registry Repo (if it doesn't exist in this region/project)
gcloud artifacts repositories create $AR_REPO --repository-format=docker --location=$REGION --description="Docker repository for my applications" --project $PROJECT_ID
# 2. Grant Logs Writer Permission to Cloud Build SA (fix for your error)
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$BUILD_SA" --role="roles/logging.logWriter" --condition=None






# --- Build, Push, and Deploy Flow (copy and paste each line in order) ---
# 3. Submit the build to Cloud Build (Builds & Pushes to Artifact Registry)
gcloud builds submit . --tag $AR_IMAGE_PATH --project $PROJECT_ID

# 4. Deploy the new image to Google Cloud Run
gcloud run deploy $SERVICE_NAME --image $AR_IMAGE_PATH --platform managed --region $REGION --allow-unauthenticated --port 8000 --project $PROJECT_ID


# If you have secrets or environment variables set on the service, you might need to re-add them. Example for a secret:
# gcloud run deploy $SERVICE_NAME --image $AR_IMAGE_PATH --platform managed --region $REGION --allow-unauthenticated --port 8000 --project $PROJECT_ID --update-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest

Write-Host "Deployment commands executed."




########################
# Before Automatization Example:
Push Docker Image:
gcloud builds submit --tag gcr.io/langgraph-chat/langgraph-api

Docker File Deploy in Google:
gcloud run deploy langgraph-api --image gcr.io/terapie-acasa/langgraph-api --platform managed --region europe-central2 --allow-unauthenticated --port 8000
