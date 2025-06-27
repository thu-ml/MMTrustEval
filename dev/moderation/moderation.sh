export OPENAI_API_KEY=

python moderation.py

# curl https://api.openai.com/v1/moderations \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "input": "I want to kill them."
#   }'
