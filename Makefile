.DEFAULT_GOAL := help
.PHONY: build

## Local

get_tf_models: ## Download and unzip tf-models
	[ -d "/tmp/model-data" ] && echo "skipping: model-data already exist\n" || \
	curl https://moon-vision-demo.s3.eu-central-1.amazonaws.com/model-data.tar.gz | tar xvz -C /tmp

build: get_tf_models ## Get TF models and build services
	docker-compose -f local.yml build

up:  ## Run server and shows logs
	docker-compose -f local.yml run --rm django ./manage.py migrate --no-input
	docker-compose -f local.yml up -d && docker-compose -f local.yml logs -f --tail=100

down:  ## Stop services
	docker-compose -f local.yml down

## Help

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
