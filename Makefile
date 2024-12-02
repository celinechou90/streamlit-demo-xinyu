BASE="nature.datahub.berkeley.edu"
MAKEFLAGS += s

.PHONY: serve
serve: 
	@echo "\n 🌎  preview at: \033[1m https://${BASE}${JUPYTERHUB_SERVICE_PREFIX}proxy/8501/ \033[0m \n"
	streamlit run app.py --server.port 8501  1> /dev/null 2>&1


.PHONY: chat 
chat: 
	@echo "\n 🌎  preview at: \033[1m https://${BASE}${JUPYTERHUB_SERVICE_PREFIX}proxy/8501/ \033[0m \n"
	streamlit run chat.py --server.port 8501  1> /dev/null 2>&1

.PHONY: rag 
rag: 
	@echo "\n 🌎  preview at: \033[1m https://${BASE}${JUPYTERHUB_SERVICE_PREFIX}proxy/8501/ \033[0m \n"
	streamlit run rag.py --server.port 8501  1> /dev/null 2>&1

.PHONY: sql 
sql: 
	@echo "\n 🌎  preview at: \033[1m https://${BASE}${JUPYTERHUB_SERVICE_PREFIX}proxy/8501/ \033[0m \n"
	streamlit run sql.py --server.port 8501  1> /dev/null 2>&1
