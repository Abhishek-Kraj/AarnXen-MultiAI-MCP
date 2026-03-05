"""Auto entity extraction — rule-based NLP for zero-cost entity/relation extraction.

Extracts technologies, people, projects, concepts, and relationships from
conversation text using regex + keyword matching. No external NLP libraries needed.
"""

import re
from typing import Optional

# Single-char or very short terms that cause excessive false positives.
# These only match when followed by specific context (e.g., "C language", "R package").
_AMBIGUOUS_SHORT_TERMS = frozenset({"c", "r", "v", "go"})


# Comprehensive tech terms — frozenset for O(1) case-insensitive lookup
_TECH_TERMS = frozenset(t.lower() for t in [
    # Languages
    "Python", "JavaScript", "TypeScript", "Rust", "Go", "Golang", "Java", "C",
    "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Elixir", "Erlang",
    "Haskell", "Clojure", "Lua", "Perl", "R", "Julia", "Dart", "Objective-C",
    "COBOL", "Fortran", "Zig", "Nim", "V", "OCaml", "F#", "Groovy", "MATLAB",
    "Assembly", "Shell", "Bash", "Zsh", "PowerShell", "SQL", "GraphQL",
    "Solidity", "Move", "Cairo", "WASM", "WebAssembly", "Mojo",
    # Frontend frameworks
    "React", "Next.js", "NextJS", "Remix", "Gatsby", "Vue", "Vue.js", "Nuxt",
    "Nuxt.js", "Angular", "Svelte", "SvelteKit", "Solid", "SolidJS", "Qwik",
    "Astro", "Ember", "Backbone", "jQuery", "Alpine.js", "AlpineJS", "HTMX",
    "Preact", "Lit", "Stencil",
    # CSS / UI
    "Tailwind", "TailwindCSS", "Bootstrap", "Chakra", "ChakraUI", "MUI",
    "Material-UI", "Ant Design", "AntDesign", "Radix", "Shadcn", "DaisyUI",
    "Bulma", "Foundation", "Sass", "SCSS", "Less", "PostCSS", "Styled-Components",
    "Emotion", "CSS Modules", "UnoCSS", "WindiCSS",
    # Backend frameworks
    "FastAPI", "Django", "Flask", "Express", "Express.js", "NestJS", "Nest.js",
    "Spring", "Spring Boot", "SpringBoot", "Rails", "Ruby on Rails", "Laravel",
    "Symfony", "Gin", "Echo", "Fiber", "Actix", "Axum", "Rocket", "Phoenix",
    "Koa", "Hapi", "Fastify", "Tornado", "Sanic", "Starlette", "Litestar",
    "ASP.NET", "Deno", "Bun", "Hono",
    # Databases
    "PostgreSQL", "Postgres", "MySQL", "MariaDB", "MongoDB", "Redis",
    "SQLite", "DynamoDB", "Cassandra", "CockroachDB", "Neo4j", "ArangoDB",
    "InfluxDB", "TimescaleDB", "ClickHouse", "Elasticsearch", "OpenSearch",
    "Solr", "Memcached", "Valkey", "Dragonfly", "ScyllaDB", "FoundationDB",
    "RocksDB", "LevelDB", "BadgerDB", "SurrealDB", "PlanetScale", "Neon",
    "Supabase", "Firebase", "Firestore", "FaunaDB", "CouchDB", "PouchDB",
    "Turso", "libSQL", "Prisma", "Drizzle", "TypeORM", "SQLAlchemy",
    "Sequelize", "Knex", "Mongoose", "Peewee", "Tortoise-ORM",
    # DevOps / Infra
    "Docker", "Kubernetes", "K8s", "Podman", "Terraform", "Pulumi", "Ansible",
    "Chef", "Puppet", "Vagrant", "Packer", "Consul", "Vault", "Nomad",
    "Nginx", "Apache", "Caddy", "Traefik", "HAProxy", "Envoy", "Istio",
    "Linkerd", "Helm", "ArgoCD", "FluxCD", "Jenkins", "CircleCI",
    "GitHub Actions", "GitLab CI", "Travis CI", "Drone", "Tekton",
    "Prometheus", "Grafana", "Datadog", "New Relic", "Sentry", "PagerDuty",
    "ELK", "Logstash", "Kibana", "Jaeger", "Zipkin", "OpenTelemetry",
    # Cloud
    "AWS", "Amazon Web Services", "GCP", "Google Cloud", "Azure",
    "Cloudflare", "Vercel", "Netlify", "Railway", "Render", "Fly.io",
    "DigitalOcean", "Linode", "Hetzner", "Vultr", "OVH", "Heroku",
    "Lambda", "EC2", "S3", "ECS", "EKS", "RDS", "SQS", "SNS",
    "CloudFront", "Route53", "IAM", "CloudWatch",
    # AI / ML
    "PyTorch", "TensorFlow", "Keras", "JAX", "scikit-learn", "sklearn",
    "Hugging Face", "HuggingFace", "LangChain", "LlamaIndex", "LangGraph",
    "CrewAI", "AutoGen", "OpenAI", "GPT", "GPT-4", "GPT-4o", "ChatGPT",
    "Claude", "Gemini", "Llama", "Mistral", "Anthropic", "Cohere", "Groq",
    "Ollama", "vLLM", "ONNX", "TensorRT", "Triton", "MLflow", "Weights & Biases",
    "WandB", "DVC", "Kubeflow", "Ray", "Dask", "Spark", "PySpark",
    "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
    "Transformers", "Diffusers", "Stable Diffusion", "Midjourney", "DALL-E",
    "Whisper", "Segment Anything", "YOLO", "OpenCV", "Pillow",
    "spaCy", "NLTK", "Gensim", "FastText", "Word2Vec",
    "XGBoost", "LightGBM", "CatBoost", "Prophet",
    "ChromaDB", "Chroma", "Pinecone", "Weaviate", "Milvus", "Qdrant",
    "FAISS", "Annoy", "pgvector",
    # Git / VCS
    "Git", "GitHub", "GitLab", "Bitbucket", "Gitea", "Forgejo",
    # Package managers / Build tools
    "npm", "yarn", "pnpm", "pip", "poetry", "uv", "conda", "cargo",
    "Maven", "Gradle", "Bazel", "CMake", "Make", "Meson", "Ninja",
    "Webpack", "Vite", "Rollup", "esbuild", "SWC", "Parcel", "Turbopack",
    "Turborepo", "Lerna", "Nx",
    # Testing
    "Jest", "Vitest", "Mocha", "Chai", "Jasmine", "Cypress", "Playwright",
    "Selenium", "Puppeteer", "pytest", "unittest", "nose", "RSpec",
    "JUnit", "TestNG", "Storybook", "Testing Library",
    # Messaging / Streaming
    "Kafka", "RabbitMQ", "NATS", "Pulsar", "ZeroMQ", "MQTT", "Redis Streams",
    "Celery", "Temporal", "Inngest",
    # Auth
    "OAuth", "JWT", "Auth0", "Clerk", "Lucia", "NextAuth", "Passport",
    "Keycloak", "SAML", "OIDC", "Okta",
    # Misc tools
    "GraphQL", "gRPC", "REST", "WebSocket", "WebRTC", "SSE",
    "Protobuf", "Avro", "Thrift", "MessagePack", "CBOR",
    "Stripe", "Plaid", "Twilio", "SendGrid", "Resend", "Postmark",
    "Algolia", "Meilisearch", "Typesense",
    "Figma", "Sketch", "Zeplin", "Storybook",
    "VS Code", "VSCode", "Neovim", "Vim", "Emacs", "IntelliJ", "WebStorm",
    "Cursor", "Zed", "Sublime Text",
    "Linux", "Ubuntu", "Debian", "CentOS", "Fedora", "Arch",
    "macOS", "Windows", "FreeBSD",
    "Nginx", "Apache", "Caddy",
    "RPC", "MCP", "LSP", "DAP",
    "Markdown", "JSON", "YAML", "TOML", "XML", "CSV", "Parquet", "Arrow",
    "Protocol Buffers",
])

# Multi-word tech terms that need special handling (space in name)
_MULTI_WORD_TERMS = {
    "ruby on rails", "spring boot", "ant design", "google cloud",
    "amazon web services", "github actions", "gitlab ci", "travis ci",
    "css modules", "stable diffusion", "segment anything", "weights & biases",
    "testing library", "redis streams", "sublime text", "vs code",
    "material-ui", "styled-components", "protocol buffers", "next.js",
    "vue.js", "nuxt.js", "nest.js", "express.js", "alpine.js", "fly.io",
}

# Relationship patterns: (regex, from_group, relation, to_group)
_RELATION_PATTERNS = [
    (r"(?:^|[\s,])(\w[\w.+-]*)\s+uses?\s+(\w[\w.+-]*)", 1, "uses", 2),
    (r"(?:^|[\s,])(\w[\w.+-]*)\s+with\s+(\w[\w.+-]*)", 1, "uses", 2),
    (r"(?:^|[\s,])(\w[\w.+-]*)\s+depends?\s+on\s+(\w[\w.+-]*)", 1, "depends_on", 2),
    (r"migrate\s+from\s+(\w[\w.+-]*)\s+to\s+(\w[\w.+-]*)", 1, "migrated_to", 2),
    (r"(\w[\w.+-]*)\s+is\s+better\s+than\s+(\w[\w.+-]*)", 1, "preferred_over", 2),
    (r"build(?:ing)?\s+(?:the\s+|a\s+|an\s+)?(\w[\w.+-]*)\s+(?:using|with|in)\s+(\w[\w.+-]*)", 1, "built_with", 2),
    (r"(\w[\w.+-]*)\s+replaces?\s+(\w[\w.+-]*)", 1, "replaces", 2),
    (r"switch(?:ing|ed)?\s+from\s+(\w[\w.+-]*)\s+to\s+(\w[\w.+-]*)", 1, "migrated_to", 2),
    (r"(\w[\w.+-]*)\s+(?:integrates?|integrated)\s+(?:with|into)\s+(\w[\w.+-]*)", 1, "integrates_with", 2),
    (r"(\w[\w.+-]*)\s+(?:runs?|running)\s+on\s+(\w[\w.+-]*)", 1, "runs_on", 2),
    (r"deploy(?:ing|ed)?\s+(?:to|on)\s+(\w[\w.+-]*)", None, "deployed_on", 1),
]

# Words that look like names but aren't
_NAME_STOPWORDS = frozenset([
    "the", "this", "that", "these", "those", "here", "there", "where", "when",
    "what", "which", "who", "whom", "whose", "how", "why", "can", "could",
    "should", "would", "will", "shall", "may", "might", "must", "need",
    "also", "just", "very", "really", "quite", "rather", "some", "any",
    "all", "each", "every", "both", "few", "many", "much", "more", "most",
    "other", "another", "such", "only", "own", "same", "than", "too",
    "now", "then", "once", "here", "there", "well", "also", "back",
    "been", "being", "have", "has", "had", "does", "did", "doing",
    "code", "review", "project", "repo", "repository", "codebase",
    "using", "about", "with", "from", "into", "like", "after", "before",
    "yes", "not", "but", "and", "for", "are", "was", "were", "let",
    "new", "old", "first", "last", "great", "good", "bad", "best", "worst",
    "sure", "okay", "thanks", "thank", "please", "help", "want", "need",
    "use", "try", "run", "set", "get", "add", "see", "look", "make",
    "hey", "hello", "dear", "regards",
])


class EntityExtractor:
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self._tech_terms = _TECH_TERMS
        self._multi_word_terms = _MULTI_WORD_TERMS
        self._relation_patterns = [
            (re.compile(pat, re.IGNORECASE), fg, rel, tg)
            for pat, fg, rel, tg in _RELATION_PATTERNS
        ]

    def extract(self, text: str) -> dict:
        if not text or len(text.strip()) < 3:
            return {"entities": [], "relations": []}

        entities = []
        seen_names = set()

        for ent in self._extract_technologies(text):
            key = ent["name"].lower()
            if key not in seen_names:
                seen_names.add(key)
                entities.append(ent)

        for ent in self._extract_people(text):
            key = ent["name"].lower()
            if key not in seen_names:
                seen_names.add(key)
                entities.append(ent)

        for ent in self._extract_projects(text):
            key = ent["name"].lower()
            if key not in seen_names:
                seen_names.add(key)
                entities.append(ent)

        for ent in self._extract_concepts(text):
            key = ent["name"].lower()
            if key not in seen_names:
                seen_names.add(key)
                entities.append(ent)

        relations = self._extract_relations(text, entities)

        return {"entities": entities, "relations": relations}

    def extract_and_store(self, text: str) -> dict:
        if not self.kb:
            return {"entities_added": 0, "entities_reinforced": 0, "relations_added": 0}

        result = self.extract(text)
        stats = {"entities_added": 0, "entities_reinforced": 0, "relations_added": 0}

        if not result["entities"] and not result["relations"]:
            return stats

        # Batch: get all existing entity names in one query
        existing_names = set()
        try:
            rows = self.kb._conn.execute(
                "SELECT LOWER(name) FROM entities"
            ).fetchall()
            existing_names = {r[0] for r in rows}
        except Exception:
            pass

        for ent in result["entities"]:
            if ent["name"].lower() in existing_names:
                stats["entities_reinforced"] += 1
            else:
                self.kb.add_entity(ent["name"], ent["type"])
                existing_names.add(ent["name"].lower())
                stats["entities_added"] += 1

        for rel in result["relations"]:
            self.kb.add_relation(rel["from"], rel["to"], rel["type"])
            stats["relations_added"] += 1

        return stats

    def _extract_technologies(self, text: str) -> list:
        found = []
        text_lower = text.lower()

        # Check multi-word terms first
        for term in self._multi_word_terms:
            if term in text_lower:
                idx = text_lower.index(term)
                original = text[idx:idx + len(term)]
                found.append({"name": _normalize_tech(original), "type": "technology"})

        # Tokenize and check single words — include dots, plus, hash for C++, C#, Next.js etc.
        for match in re.finditer(r'[A-Za-z][A-Za-z0-9.+#-]*', text):
            token = match.group()
            token_lower = token.lower()
            if token_lower in self._tech_terms:
                # Skip ambiguous short terms unless they have qualifying context
                if token_lower in _AMBIGUOUS_SHORT_TERMS:
                    if not _has_tech_context(text, match.start(), match.end(), token_lower):
                        continue
                name = _normalize_tech(token)
                if not any(f["name"].lower() == name.lower() for f in found):
                    found.append({"name": name, "type": "technology"})

        return found

    def _extract_people(self, text: str) -> list:
        found = []
        seen = set()

        # Pattern: @mentions
        for match in re.finditer(r'@(\w+)', text):
            name = match.group(1)
            if name.lower() not in seen and name.lower() not in _NAME_STOPWORDS:
                seen.add(name.lower())
                found.append({"name": name, "type": "person"})

        # Pattern: capitalized word after "by", "from", "with" (at word boundary)
        for match in re.finditer(r'\b(?:by|from|with|author|creator|maintainer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text):
            name = match.group(1).strip()
            name_lower = name.lower()
            if (name_lower not in seen
                    and name_lower not in _NAME_STOPWORDS
                    and name_lower not in self._tech_terms):
                seen.add(name_lower)
                found.append({"name": name, "type": "person"})

        return found

    def _extract_projects(self, text: str) -> list:
        found = []
        seen = set()

        # GitHub URLs: github.com/org/repo
        for match in re.finditer(r'github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)', text):
            repo = match.group(1).rstrip("/.")
            if repo.lower() not in seen:
                seen.add(repo.lower())
                found.append({"name": repo, "type": "project"})

        # Pattern: "project/repo/repository/codebase X"
        for match in re.finditer(r'\b(?:project|repo|repository|codebase)\s+([A-Za-z][A-Za-z0-9_.-]+)', text, re.IGNORECASE):
            name = match.group(1).strip()
            name_lower = name.lower()
            if (name_lower not in seen
                    and name_lower not in _NAME_STOPWORDS
                    and name_lower not in self._tech_terms):
                seen.add(name_lower)
                found.append({"name": name, "type": "project"})

        return found

    def _extract_concepts(self, text: str) -> list:
        found = []
        seen = set()

        # Pattern: "about/regarding/related to/for <phrase>"
        for match in re.finditer(r'\b(?:about|regarding|related\s+to)\s+([a-z][a-z\s]{2,30}?)(?:[.,;!?]|\s+(?:and|but|or|which|that|is|are|was|were|with|using|in|on|at|to|for|the)\b)', text, re.IGNORECASE):
            phrase = match.group(1).strip()
            words = phrase.split()
            if len(words) > 3:
                words = words[:3]
            concept = " ".join(words)
            concept_lower = concept.lower()
            if (concept_lower not in seen
                    and concept_lower not in _NAME_STOPWORDS
                    and concept_lower not in self._tech_terms
                    and len(concept) > 2):
                seen.add(concept_lower)
                found.append({"name": concept, "type": "concept"})

        return found

    def _extract_relations(self, text: str, entities: list) -> list:
        found = []
        entity_names = {e["name"].lower() for e in entities}

        for pattern, from_group, rel_type, to_group in self._relation_patterns:
            for match in pattern.finditer(text):
                if from_group is not None:
                    from_name = match.group(from_group)
                else:
                    from_name = None
                to_name = match.group(to_group)

                # Only create relations where at least the target is a known entity
                to_lower = to_name.lower()
                if to_lower in entity_names or to_lower in self._tech_terms:
                    to_canonical = _find_canonical(to_name, entities)
                    if from_name:
                        from_lower = from_name.lower()
                        if from_lower in entity_names or from_lower in self._tech_terms:
                            from_canonical = _find_canonical(from_name, entities)
                            if from_canonical.lower() != to_canonical.lower():
                                rel = {"from": from_canonical, "to": to_canonical, "type": rel_type}
                                if rel not in found:
                                    found.append(rel)

        return found


_TECH_CONTEXT_WORDS = frozenset({
    "language", "programming", "code", "compiler", "library", "package",
    "framework", "runtime", "syntax", "function", "module", "script",
    "codebase", "project", "build", "compile", "import", "version",
})


def _has_tech_context(text: str, start: int, end: int, term: str) -> bool:
    """Check if an ambiguous short term (C, R, V, Go) has programming context nearby."""
    # Check 60 chars before and after
    window_start = max(0, start - 60)
    window_end = min(len(text), end + 60)
    window = text[window_start:window_end].lower()
    for ctx_word in _TECH_CONTEXT_WORDS:
        if ctx_word in window:
            return True
    # Special patterns: "in C", "using C", "written in Go"
    before = text[max(0, start - 15):start].lower().strip()
    if before.endswith(("in", "using", "with", "of", "the")):
        return True
    return False


def _normalize_tech(token: str) -> str:
    """Normalize tech name casing for consistency."""
    lookup = {
        "python": "Python", "javascript": "JavaScript", "typescript": "TypeScript",
        "rust": "Rust", "golang": "Go", "java": "Java",
        "c++": "C++", "c#": "C#", "ruby": "Ruby", "php": "PHP",
        "swift": "Swift", "kotlin": "Kotlin", "scala": "Scala",
        "react": "React", "vue": "Vue", "angular": "Angular",
        "svelte": "Svelte", "nextjs": "Next.js", "next.js": "Next.js",
        "nuxt": "Nuxt", "nuxt.js": "Nuxt.js", "nest.js": "NestJS",
        "nestjs": "NestJS", "express": "Express", "express.js": "Express",
        "fastapi": "FastAPI", "django": "Django", "flask": "Flask",
        "fastify": "Fastify", "spring": "Spring", "spring boot": "Spring Boot",
        "springboot": "Spring Boot", "rails": "Rails", "ruby on rails": "Ruby on Rails",
        "laravel": "Laravel", "gin": "Gin", "fiber": "Fiber",
        "actix": "Actix", "axum": "Axum", "phoenix": "Phoenix",
        "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
        "mysql": "MySQL", "mongodb": "MongoDB", "redis": "Redis",
        "sqlite": "SQLite", "dynamodb": "DynamoDB", "cassandra": "Cassandra",
        "elasticsearch": "Elasticsearch", "neo4j": "Neo4j",
        "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
        "terraform": "Terraform", "ansible": "Ansible",
        "nginx": "Nginx", "apache": "Apache", "caddy": "Caddy",
        "aws": "AWS", "gcp": "GCP", "azure": "Azure",
        "vercel": "Vercel", "netlify": "Netlify", "heroku": "Heroku",
        "cloudflare": "Cloudflare",
        "pytorch": "PyTorch", "tensorflow": "TensorFlow", "keras": "Keras",
        "jax": "JAX", "scikit-learn": "scikit-learn", "sklearn": "scikit-learn",
        "langchain": "LangChain", "llamaindex": "LlamaIndex",
        "openai": "OpenAI", "claude": "Claude", "gemini": "Gemini",
        "llama": "Llama", "mistral": "Mistral", "anthropic": "Anthropic",
        "groq": "Groq", "ollama": "Ollama",
        "git": "Git", "github": "GitHub", "gitlab": "GitLab",
        "npm": "npm", "yarn": "yarn", "pnpm": "pnpm", "pip": "pip",
        "poetry": "Poetry", "uv": "uv", "cargo": "Cargo",
        "webpack": "Webpack", "vite": "Vite", "esbuild": "esbuild",
        "jest": "Jest", "vitest": "Vitest", "pytest": "pytest",
        "playwright": "Playwright", "cypress": "Cypress",
        "kafka": "Kafka", "rabbitmq": "RabbitMQ", "nats": "NATS",
        "graphql": "GraphQL", "grpc": "gRPC", "rest": "REST",
        "jwt": "JWT", "oauth": "OAuth",
        "prisma": "Prisma", "drizzle": "Drizzle", "sqlalchemy": "SQLAlchemy",
        "tailwind": "Tailwind", "tailwindcss": "Tailwind",
        "bootstrap": "Bootstrap", "shadcn": "Shadcn",
        "supabase": "Supabase", "firebase": "Firebase",
        "stripe": "Stripe", "twilio": "Twilio",
        "linux": "Linux", "ubuntu": "Ubuntu", "macos": "macOS",
        "windows": "Windows",
        "vscode": "VS Code", "vs code": "VS Code",
        "neovim": "Neovim", "vim": "Vim",
        "mcp": "MCP", "lsp": "LSP",
        "pandas": "Pandas", "numpy": "NumPy", "scipy": "SciPy",
        "matplotlib": "Matplotlib",
        "chromadb": "ChromaDB", "chroma": "ChromaDB",
        "pinecone": "Pinecone", "weaviate": "Weaviate", "qdrant": "Qdrant",
        "faiss": "FAISS", "pgvector": "pgvector",
        "celery": "Celery", "temporal": "Temporal",
        "prometheus": "Prometheus", "grafana": "Grafana", "sentry": "Sentry",
        "starlette": "Starlette", "litestar": "Litestar",
        "deno": "Deno", "bun": "Bun", "hono": "Hono",
    }
    return lookup.get(token.lower(), token)


def _find_canonical(name: str, entities: list) -> str:
    """Find the canonical (properly-cased) name from the entity list."""
    for ent in entities:
        if ent["name"].lower() == name.lower():
            return ent["name"]
    return _normalize_tech(name)
