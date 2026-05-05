"""
Centralized configuration for the IEEE DataPort Bedrock system.

This module contains all configuration constants, thresholds, and settings
used throughout the system, providing a single source of truth for all
configuration values.
"""

import os
from dataclasses import dataclass

# Auto-load .env file if present (for local development)
# This allows running scripts directly without manual export commands
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables


@dataclass
class PipelineConfig:
    """
    Runtime configuration for the metadata generation pipeline.

    Attributes:
        analysis_bucket: S3 bucket where analysis results are stored
        dataport_url: URL to the dataset on IEEE Dataport
        output_metadata_path: S3 path for saving metadata files
        zipped_synthesis_base: Filename for synthesis files (constant)
    """

    analysis_bucket: str
    dataport_url: str
    output_metadata_path: str
    zipped_synthesis_base: str


# =============================================================================
# LLM Configuration
# =============================================================================

# Default model keys for different operations
# Can be overridden via SONNET_MODEL_KEY env var (e.g., "claudeSonnet4_6")
_SONNET_MODEL_KEY = os.environ.get("SONNET_MODEL_KEY", "claudeSonnet4_6")
DEFAULT_MODEL_KEY = _SONNET_MODEL_KEY
ANALYSIS_MODEL_KEY = _SONNET_MODEL_KEY

# Model key for the initial visualization code generation pass.
# Defaults to Sonnet 4.6; the agentic fix loop and all other calls use DEFAULT_MODEL_KEY.
INITIAL_VIZ_MODEL_KEY = os.environ.get("INITIAL_VIZ_MODEL_KEY", "claudeSonnet4_6")

# Fallback model ID when MODELS dict lookup fails
# Used by chat_lambda and chat_tester as a fallback if the key doesn't exist
DEFAULT_FALLBACK_MODEL_ID = "anthropic.claude-sonnet-4-6-20261001"

# =============================================================================
# Chat Response Configuration
# =============================================================================

# Max tokens for response generation
# Using unified default of 16384 to avoid truncation issues
# Previous two-tier approach (8192 default / 16384 for code) had detection gaps
DEFAULT_MAX_TOKENS = 16384

# Chat-specific max tokens — lower than pipeline to reduce TPM reservation.
# Bedrock reserves (input_tokens + max_tokens) from TPM at request start.
# CloudWatch p99.5 = 10,176 output tokens; auto-continuation handles any truncation.
CHAT_MAX_TOKENS = 8192

# Chat context budget — limits total input tokens to prevent TPM exhaustion.
# Bedrock's 5x output burndown means each request reserves input + (max_tokens × 5) TPM.
# With 50K budget: input ≈ 35K → TPM ≈ 76K per request (vs 91K+ uncapped at 200K).
# The ContextManager's sliding window prunes middle history to fit this budget
# while preserving the first user message (intent) and last 5 exchange pairs.
CHAT_MAX_CONTEXT_TOKENS = 50000

# History summarization — use Haiku for speed and cost efficiency.
# Generates a 2-3 sentence summary of dropped messages (~1s, ~$0.001/call).
HISTORY_SUMMARIZATION_MODEL_KEY = "claudeHaiku4_5"
HISTORY_SUMMARIZATION_MAX_TOKENS = 300

# Tier-based immediate context size — how many recent exchange pairs
# are always protected from pruning. Larger datasets have bigger manifests,
# so fewer exchanges fit in the 50K budget.
CHAT_IMMEDIATE_CONTEXT_SIZE = {
    "SMALL": 5,    # ~10 messages protected
    "MEDIUM": 4,   # ~8 messages protected
    "LARGE": 3,    # ~6 messages protected
}
CHAT_IMMEDIATE_CONTEXT_SIZE_DEFAULT = 5

# Maximum continuation attempts for truncated responses
MAX_CONTINUATION_ATTEMPTS = 2

# Maximum tool use iterations (prevent infinite loops)
# Increased from 5 to 8 to give LLM more recovery attempts after errors
MAX_TOOL_USE_ITERATIONS = 8

# Tool loop time limit (seconds) - break before upstream proxy timeout (~60s)
TOOL_LOOP_TIME_LIMIT_SECONDS = 55

# Sync handler Bedrock read timeout (seconds).
# Must be shorter than the upstream proxy timeout (~60s) so the handler
# can catch the timeout and return a helpful error instead of a raw 504.
# Budget: ~5s handler setup + 45s Bedrock + ~5s response formatting = ~55s
SYNC_BEDROCK_READ_TIMEOUT = 45

# Bedrock model pricing per million tokens (Sonnet 4.5)
MODEL_INPUT_COST_PER_MILLION = 3.0
MODEL_OUTPUT_COST_PER_MILLION = 15.0

# =============================================================================
# Streaming Configuration
# =============================================================================

# Streaming response settings for Lambda Web Adapter + FastAPI
STREAMING_BUFFER_SIZE = 0  # Emit every token immediately (0 = no buffering)

# Streaming resilience settings for graceful failure handling
# Heartbeat interval during long operations (seconds)
# Client should timeout if no event received within 2-3x this value
STREAMING_HEARTBEAT_INTERVAL_SECONDS = 10

# Maximum streaming response time (seconds) - after this, stream is terminated
STREAMING_MAX_RESPONSE_SECONDS = 300  # 5 minutes

# =============================================================================
# Bedrock Client Configuration
# =============================================================================

# Unified timeout settings for Bedrock clients
BEDROCK_READ_TIMEOUT = 600  # 10 minutes - allows for long-running inference
BEDROCK_CONNECT_TIMEOUT = 10  # Connection timeout
BEDROCK_MAX_RETRIES = 3  # Standard retry attempts (used by pipeline)

# Chat-specific retry limit — lower than pipeline to prevent retry amplification.
# When Bedrock is throttled, boto3's standard retry mode retries 3 times with
# exponential backoff (~7s total delay). For chat, this wastes time since the
# frontend has its own fallback logic (streaming → sync).
# max_attempts=2 means one automatic retry — enough to recover from transient
# 500s but limits throttle cascade delay to ~2s instead of ~7s.
CHAT_BEDROCK_MAX_RETRIES = 2

# Seconds the frontend should wait before retrying a throttled request.
# Sent in throttle error responses from both streaming and sync handlers.
THROTTLE_RETRY_AFTER_SECONDS = 5

# Anthropic beta flags for extended context windows
# This flag enables 1M context support for Sonnet 4.5
ANTHROPIC_1M_CONTEXT_BETA_FLAG = "context-1m-2025-08-07"

# Prompt caching beta flag for Bedrock
# Enables system prompt caching to reduce latency and costs
ANTHROPIC_PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"

# =============================================================================
# Bedrock Guardrails Configuration
# =============================================================================

# Guardrail for topic adherence, PII masking, and prompt attack prevention
# IEEE DataPort Research Guardrail - enforces data science focus
GUARDRAIL_IDENTIFIER = "xr2jjgm7t06o"
GUARDRAIL_VERSION = "DRAFT"  # Use specific version number in production
GUARDRAIL_ENABLED = True  # Toggle to disable guardrails (e.g., for testing)
GUARDRAIL_TRACE_ENABLED = True  # Enable trace for debugging blocked prompts

# Standardized refusal message when guardrail intervenes
GUARDRAIL_REFUSAL_MESSAGE = (
    "I am designed to assist with data analysis and research for this specific dataset. "
    "I cannot answer general knowledge or creative requests."
)

# Token limits and chunking thresholds
SONNET_4_5_CONTEXT_WINDOW = 1000000  # Claude Sonnet 4.5 hard model limit
HAIKU_CHUNK_LIMIT = 180000  # Claude 3.5 Haiku (200K context - 20K buffer for output)
SONNET_CHUNK_LIMIT = 200000  # Claude Sonnet 4.0 token limit
SONNET_4_5_CHUNK_LIMIT = (
    900000  # Claude Sonnet 4.5 (1M context - 100K buffer for output)
)
SONNET_4_6_CHUNK_LIMIT = (
    900000  # Claude Sonnet 4.6 (1M context - 100K buffer for output)
)

# Adaptive chunking thresholds for structured files
ULTRA_WIDE_COLUMN_THRESHOLD = 1500
WIDE_COLUMN_THRESHOLD = 500
STANDARD_COLUMN_THRESHOLD = 100

# Chat manifest optimization thresholds
MANIFEST_WIDE_DATASET_THRESHOLD = 30  # Columns above this trigger pattern grouping
MANIFEST_MAX_FILES = 15  # Maximum files to include in chat manifest schema
MANIFEST_MAX_COLUMNS_PER_FILE = 15  # Max columns shown per file before grouping
NARRATIVE_CONDENSATION_THRESHOLD = 5000  # Characters above this trigger LLM condensation

# =============================================================================
# Exploration Paths Configuration
# =============================================================================

# Expected audience distribution for exploration paths (order matters)
# Format: 2 general (accessible), 1 methodology (data handling), 2 technical (research)
EXPLORATION_PATH_AUDIENCES = ["general", "general", "methodology", "technical", "technical"]

# Total expected path count (derived from audience list length)
EXPLORATION_PATH_COUNT = len(EXPLORATION_PATH_AUDIENCES)

# =============================================================================
# Inventory Tier Configuration
# =============================================================================

# File count thresholds for tier classification
INVENTORY_TIER_SMALL_MAX = 15       # Full inline (current behavior)
INVENTORY_TIER_MEDIUM_MAX = 100     # Condensed inline (all files)
# Above MEDIUM_MAX → LARGE with pattern grouping + tool access

# Tier names for meta.tier field
TIER_SMALL = "SMALL"
TIER_MEDIUM = "MEDIUM"
TIER_LARGE = "LARGE"

# Inventory strategy names for meta.inventory_strategy field
STRATEGY_FULL_INLINE = "full_inline"
STRATEGY_CONDENSED_INLINE = "condensed_inline"
STRATEGY_PATTERN_GROUPED = "pattern_grouped"

# Per-tier file limits for manifest
MANIFEST_MAX_FILES_SMALL = 15       # Current behavior
MANIFEST_MAX_FILES_MEDIUM = 100     # All files, condensed detail
MANIFEST_MAX_FILES_LARGE = 15       # Priority files only

# File sampling limits for large datasets (metadata generation)
FILE_SAMPLING_THRESHOLD = 500       # Datasets larger than this get sampled
FILE_SAMPLING_TARGET = 500          # Target sample size for initial sampling
FILE_REMAINING_SAMPLE_MAX = 100     # Max remaining files in synthesis output

# Adaptive media metadata extraction (audio/video Range-then-fallback)
# During the probe phase, every media file is extracted.  If the fallback rate
# exceeds the threshold, subsequent files are extracted only every Nth file.
MEDIA_PROBE_SIZE = 10               # Files to test before deciding strategy
MEDIA_FALLBACK_THRESHOLD = 0.5      # Fallback rate that triggers sampling mode
MEDIA_SAMPLE_INTERVAL = 10          # In sampling mode, extract every Nth file
MEDIA_SAMPLE_CAP = 50               # Max total extracted files per extension

# Pattern grouping configuration (for LARGE tier)
FILE_PATTERN_MIN_GROUP_SIZE = 5     # Min files to form a pattern group
FILE_PATTERN_EXAMPLE_COUNT = 3      # Example files shown per pattern
FILE_PATTERN_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.csv', '.json', '.txt', '.xml'}

# Directory tree configuration
DIRECTORY_TREE_MAX_LINES = 100      # Max lines for directory tree in manifest
DIRECTORY_TREE_COLLAPSE_THRESHOLD = 10  # Collapse dirs with more subdirs than this

# Directory analysis thresholds
MASSIVE_DIRECTORY_THRESHOLD = 200
LARGE_DIRECTORY_THRESHOLD = 50

# =============================================================================
# Model Configurations
# =============================================================================

# AWS Account ID loaded from environment (required for inference profile ARNs)
# Fail fast if not set - invalid ARNs are hard to debug
_AWS_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID")
if not _AWS_ACCOUNT_ID:
    raise EnvironmentError(
        "AWS_ACCOUNT_ID environment variable is required for Bedrock model ARNs. "
        "Set it to your 12-digit AWS account ID."
    )

MODELS = {
    "claudeHaiku4_5": f"arn:aws:bedrock:us-east-1:{_AWS_ACCOUNT_ID}:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claudeSonnet4_0": f"arn:aws:bedrock:us-east-1:{_AWS_ACCOUNT_ID}:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0",
    "claudeSonnet4_5": f"arn:aws:bedrock:us-east-1:{_AWS_ACCOUNT_ID}:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claudeSonnet4_6": f"arn:aws:bedrock:us-east-1:{_AWS_ACCOUNT_ID}:inference-profile/global.anthropic.claude-sonnet-4-6",
}

# Model ID substrings that support 1M token extended context via beta header.
# Update this set when new models gain extended context support.
EXTENDED_CONTEXT_MODEL_TAGS = {"sonnet-4-5", "sonnet-4-6"}

# =============================================================================
# File Processing Configuration
# =============================================================================

# Supported file extensions for different data types
STRUCTURED_EXTENSIONS = {".csv", ".json", ".tsv", ".txt", ".parquet", ".xlsx", ".mat", ".dta"}

# Excel workbook processing
EXCEL_INCLUDE_HIDDEN_SHEETS = False  # Whether to process hidden Excel worksheets
UNSTRUCTURED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
COMPRESSED_EXTENSIONS = {".zip"}

# File size and processing limits
MAX_FILE_SIZE_MB = 100
MAX_CONTENT_LENGTH = 1000000  # 1MB text content limit

# Canonical media extension sets — import these instead of defining locally
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    ".svg", ".ico", ".heic", ".heif", ".raw", ".cr2", ".nef", ".dng",
}
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aiff", ".opus", ".aac",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".mpg", ".mpeg",
}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Binary file extensions (not text-parseable)
# These files are handled separately from text-based analysis
BINARY_EXTENSIONS = {
    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".svg",
    ".ico",
    ".heic",
    ".heif",
    ".raw",
    ".cr2",
    ".nef",
    ".dng",
    # Archives (included despite pre-unzip, as some datasets still contain them)
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    # Audio
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".wma",
    ".opus",
    # Video
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    # Documents (binary formats)
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    # Binaries
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".pyc",
    # Network packet captures
    ".pcap",
    ".pcapng",
    ".cap",
    ".dmp",
    ".5vw",
    ".trc",
    ".snoop",
    ".atc",
    # Scientific/disk images and data formats
    ".img",
    ".hdr",
    ".iso",
    ".dmg",
    ".vmdk",
    ".vdi",
    ".qcow2",
    ".obj",
    ".stl",
    ".ply",  # 3D model formats
    ".grib",
    ".grib2",
    ".nc",
    ".hdf",
    ".hdf5",
    ".h5",  # Scientific data
    ".npz",
    ".npy",  # NumPy array formats
    ".pkl",
    ".pickle",  # Python serialized objects
    ".nii",
    ".nii.gz",
    ".dcm",
    ".dicom",  # Medical imaging
    ".fits",
    ".fit",  # Astronomical data
    # Specialized binary formats
    ".cf32",  # Complex float 32 (SDR/RF signal data)
    ".dgs",  # Digisonde ionogram data
    ".grd",  # Grid data (various scientific tools)
}

# Extensions that load_structured_data() can actually parse into DataFrames.
# Used by load_data_from_s3() to skip S3 download for non-loadable file types
# (e.g., video, audio, raw binary) that would otherwise be downloaded only to
# return None after falling through all parser branches.
# Update this set when adding a new parser to load_structured_data().
STRUCTURED_LOADABLE_EXTENSIONS = {
    ".mat",  # MATLAB (scipy/h5py)
    ".dat",  # Binary data (custom parser)
    ".bin",  # Binary data (custom parser)
    ".csv", ".json", ".jsonl", ".tsv", ".txt",  # Text-based tabular
    ".parquet",  # Columnar storage
    ".xlsx", ".xls",  # Excel
    ".dta",  # Stata (pd.read_stata, added by PR #106)
    ".html",  # Treated as unstructured documentation
    ".mlx",  # MATLAB Live Script (returns text summary)
}

# Large text file sampling threshold
# Files exceeding this size are analyzed via sampling (S3 Range requests)
# rather than full download. Applies to text files only (.txt, .log, .csv, etc.)
LARGE_TEXT_FILE_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100MB

# Sample chunk sizes for large text file analysis (S3 Range requests)
# Total sample: ~10MB for files exceeding LARGE_TEXT_FILE_THRESHOLD_BYTES
# Values in bytes - can be tuned based on performance/accuracy tradeoffs
LARGE_TEXT_SAMPLE_SIZES = {
    "beginning": 3 * 1024 * 1024,  # 3MiB - headers, initial content, encoding detection
    "middle": 4 * 1024 * 1024,  # 4MiB - representative content from file center
    "end": 3 * 1024 * 1024,  # 3MiB - closing content, summaries
}

# Large file size guard for visualization data loading.
# Files exceeding their format's threshold get special handling in load_data_from_s3():
# - Text-based formats (CSV, TSV, JSON, etc.): download but parse only first N rows
# - Binary formats (Excel, .mat): skip entirely (return None)
# This prevents timeout/OOM when viz code tries to load huge files into DataFrames.
LARGE_FILE_DEFAULT_THRESHOLD_BYTES = 200 * 1024 * 1024  # 200MB

# Per-format overrides. Parquet is columnar and memory-efficient; it can handle
# larger files than row-oriented text formats without OOM risk.
LARGE_FILE_THRESHOLD_OVERRIDES = {
    ".parquet": 500 * 1024 * 1024,  # 500MB
}

# Backward-compatible alias.
LARGE_FILE_SKIP_THRESHOLD_BYTES = LARGE_FILE_DEFAULT_THRESHOLD_BYTES
LARGE_FILE_SAMPLE_NROWS = 10_000  # Max rows to parse from oversized text files

# Text-based extensions eligible for nrows sampling (subset of STRUCTURED_LOADABLE_EXTENSIONS)
LARGE_FILE_SAMPLABLE_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".txt"}

# When multiple files referenced in generated code have a combined size exceeding
# this threshold, the agentic loop warns the LLM that sequential loading will
# likely exceed the execution timeout.  Individual files may be under the per-file
# threshold above, but their cumulative download time is still problematic.
CUMULATIVE_LOAD_WARNING_BYTES = 300 * 1024 * 1024  # 300MB

# =============================================================================
# S3 Configuration
# =============================================================================

# Content types for S3 uploads
DEFAULT_CONTENT_TYPE = "application/json"
MARKDOWN_CONTENT_TYPE = "text/markdown; charset=utf-8"
PYTHON_CONTENT_TYPE = "text/x-python"
VISUALIZATION_CONTENT_TYPE = "image/png"

# S3 path configurations
ANALYSIS_DIR = "_data_analysis_ai_"
METADATA_DIR = "_metadata_"
RESULTS_DIR = "_analysis_results_"
VISUALIZATIONS_DIR = "visualizations"

# Interactive chat S3 paths (Phase 2 chat system)
# Using _analysis_results_ for backward compatibility with Phase 1 outputs
# This ensures existing datasets continue to work without migration
INTERACTIVE_CHAT_DIR = "_analysis_results_"
CHAT_MANIFEST_NAME = "chat_manifest.json"
FULL_INVENTORY_NAME = "full_file_inventory.json"

# S3 file naming conventions
ZIP_SYNTHESIS_NAME = "synthesis_for_llm.json"
STRUCTURED_SYNTHESIS_NAME = "structured_data_synthesis_for_llm.json"
ANALYSIS_FILE_NAME = "analysis_report.md"
PYTHON_CODE_NAME = "executable_code.py"
AMALGAMATED_CONTEXT_NAME = "amalgamated_context_summary.json"
DATASET_IDENTIFIER = "dataset_identifier.py"
TOKEN_USAGE_FILE = "_token_usage.json"

# =============================================================================
# Data Processing Configuration
# =============================================================================

# Column and data thresholds
MAX_CATEGORICAL_CATEGORIES = 50

# Headerless data detection thresholds
# Used to detect when data rows are incorrectly interpreted as column headers
HEADERLESS_MAX_COLUMN_NAME_LENGTH = 100  # Characters - longer suggests data as header
HEADERLESS_MAX_COLUMN_WORDS = 10  # Words - more suggests data as header
MAX_LABEL_LENGTH = 20
MAX_VARIABLES_FOR_PLOTTING = 5

# Sampling and chunking parameters
DEFAULT_SAMPLE_SIZE = 15
MICRO_CHUNK_SIZE = 25
MAX_MICRO_CHUNKS = 8
HYBRID_CHUNK_TOKEN_LIMIT = 50000

# Data cleaning parameters
UNNAMED_COLUMN_PATTERNS = ["Unnamed:", "Unnamed "]
EMPTY_COLUMN_THRESHOLD = 0.95  # 95% empty values

# Text analysis parameters
TOP_TOKENS_FOR_FREQUENCY_ANALYSIS = (
    30  # Number of top words/characters in frequency analysis
)

# =============================================================================
# Visualization Configuration
# =============================================================================

# Max output tokens for visualization code generation.
# Higher limit reduces truncation at the source; cases that still truncate
# are caught by the agentic fix loop (Layer 1).
VIZ_CODE_MAX_TOKENS = 32768

# Matplotlib configuration
MATPLOTLIB_BACKEND = "Agg"
DEFAULT_FONT_FAMILY = "DejaVu Sans"
DEFAULT_FONT_SIZE = 10
DEFAULT_FIGURE_DPI = 100

# Non-Latin script font paths (relative to project root)
# These fonts support rendering text in various scripts/languages
DEVANAGARI_FONT_PATH = "lib/visualization/fonts/NotoSansDevanagari-Regular.ttf"
CJK_FONT_PATH = "lib/visualization/fonts/NotoSansCJKsc-Regular.ttf"
ARABIC_FONT_PATH = "lib/visualization/fonts/NotoSansArabic-Regular.ttf"
BENGALI_FONT_PATH = "lib/visualization/fonts/NotoSansBengali-Regular.ttf"
THAI_FONT_PATH = "lib/visualization/fonts/NotoSansThai-Regular.ttf"
TAMIL_FONT_PATH = "lib/visualization/fonts/NotoSansTamil-Regular.ttf"
HEBREW_FONT_PATH = "lib/visualization/fonts/NotoSansHebrew-Regular.ttf"

# Visualization limits and thresholds
MAX_CORRELATION_VARIABLES = 15
MAX_SIGNAL_PLOTS = 6
MAX_DISTRIBUTION_PLOTS = 8
MIN_CORRELATION_THRESHOLD = 0.3

# Label calculation constants for dynamic category display
PIXELS_PER_INCH = 80  # Display resolution for label spacing calculations
PIXELS_PER_CHAR = 7  # Average character width in pixels
LABEL_PADDING_FACTOR = 1.5  # Spacing factor between labels
MIN_DISPLAY_CATEGORIES = 10  # Minimum categories to show in plots
MAX_DISPLAY_CATEGORIES = 30  # Maximum categories to show in plots
DEFAULT_MAX_CATEGORIES = 15  # Fallback when no labels provided
DEFAULT_LABEL_MAX_LENGTH = 20  # Maximum characters before truncating labels

# Category aggregation for limit_to_top_n_categories()
CATEGORY_AGGREGATE_LABEL = "Other"  # Label for aggregated minor categories

# Category visualization tier thresholds
# Used in both LLM prompt guidance AND runtime get_display_limit_for_classes()
CATEGORY_TIER_1_THRESHOLD = 20  # Above this, recommend/apply category limiting
CATEGORY_TIER_2_THRESHOLD = 50  # Above this, recommend different viz type

# Display limits for get_display_limit_for_classes()
# How many categories to actually show at each tier
# fmt: off
DISPLAY_TIER_1_LIMIT = 10  # Show when MIN_DISPLAY_CATEGORIES < classes <= CATEGORY_TIER_1_THRESHOLD
DISPLAY_TIER_2_LIMIT = 15  # Show when CATEGORY_TIER_1_THRESHOLD < classes <= CATEGORY_TIER_2_THRESHOLD
DISPLAY_TIER_DEFAULT_LIMIT = 20  # Show when classes > CATEGORY_TIER_2_THRESHOLD
# fmt: on
TIMESTAMP_CONFIDENCE_THRESHOLD = (
    0.8  # Confidence threshold (80%) for timestamp detection
)
TIMESTAMP_SAMPLE_SIZE = 5  # Number of values to sample for timestamp detection

# Color schemes and styling
DEFAULT_COLORMAP = "viridis"
CORRELATION_COLORMAP = "RdBu_r"
SIGNAL_COLORS = ["blue", "red", "green", "orange", "purple", "brown"]

# =============================================================================
# Pie Chart Quality Control (AST-based auto-fix settings)
# =============================================================================
# These settings are applied by the AST transformer to ensure consistent,
# readable pie charts regardless of LLM-generated code variations.

# Autotext (percentage labels inside slices) styling
PIE_AUTOTEXT_FONTSIZE = 9
PIE_AUTOTEXT_FONTWEIGHT = "bold"
PIE_AUTOTEXT_COLOR = "white"

# Slice label (category names outside slices) styling
PIE_LABEL_FONTSIZE = 10
PIE_LABEL_FONTWEIGHT = "normal"

# Layout parameters (distances are relative: 0=center, 1=edge)
PIE_PCTDISTANCE = 0.80  # Distance of % label from center (inside slice, near edge)
PIE_LABELDISTANCE = 1.1  # Distance of slice label from center (>1 = outside edge)
PIE_STARTANGLE = 90  # Rotation of first slice (90 = start at 12 o'clock)
PIE_SHADOW = False  # Shadow effect (disabled for clarity)

# Wedge (slice) styling
PIE_WEDGE_EDGECOLOR = "black"
PIE_WEDGE_LINEWIDTH = 1.5

# Visibility threshold - hide % labels for slices smaller than this
PIE_AUTOPCT_MIN_PERCENT = 3.0

# Category limit - maximum slices before recommending bar chart
PIE_MAX_SLICES = 8

# Legend settings (auto-injected when LLM doesn't create one)
PIE_LEGEND_ENABLED = True  # Whether to auto-inject legends for pie charts
PIE_LEGEND_LOC = "center left"  # Legend location anchor point
PIE_LEGEND_BBOX_TO_ANCHOR = (
    1.15,
    0.5,
)  # Position legend to the right of pie (further out to avoid label overlap)
PIE_LEGEND_FONTSIZE = 10  # Legend text font size

# =============================================================================
# Bar Chart Quality Control (AST-based auto-fix settings)
# =============================================================================
# These settings ensure consistent, readable bar charts.

# Maximum categories to display (excess are truncated via .head())
BAR_MAX_CATEGORIES = 25

# Annotation styling for count/percentage labels on bars
BAR_ANNOTATION_FONTSIZE = 9
BAR_ANNOTATION_FONTWEIGHT = "normal"

# X-axis label rotation for congested charts
BAR_XTICK_ROTATION = 45

# =============================================================================
# Plotly Interactive Chart Configuration
# =============================================================================
# These settings control Plotly chart defaults to prevent text overlap.

# Default margins (left, right, top, bottom) in pixels
PLOTLY_DEFAULT_MARGIN = {"l": 60, "r": 60, "t": 60, "b": 100}

# Default font size for chart text
PLOTLY_DEFAULT_FONT_SIZE = 11

# Default x-axis tick rotation angle (degrees, negative = clockwise)
PLOTLY_TICK_ANGLE = -45

# Standoff distance between axis title and axis (pixels)
PLOTLY_TITLE_STANDOFF = 15

# =============================================================================
# AST Code Refinement Configuration
# =============================================================================
# These settings control the AST-based post-processing of LLM-generated code.

# Minimum line retention ratio after AST processing
# ast.unparse() legitimately strips comments and blank lines, so 30-40% loss is normal.
# Only flag as corruption if line count drops below this ratio (0.5 = 50% retained).
AST_MIN_LINE_RETENTION = 0.5

# =============================================================================
# Logging Configuration
# =============================================================================

# Logging levels and formats
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# Error Handling Configuration
# =============================================================================

# Retry configuration
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5
EXPONENTIAL_BACKOFF = True

# Timeout settings
DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes
LLM_TIMEOUT_SECONDS = 300  # 5 minutes
S3_TIMEOUT_SECONDS = 120  # 2 minutes

# =============================================================================
# Phase 2.1: Code Execution Configuration (Agentic Tools)
# =============================================================================

# Feature flag for code execution tools
CODE_EXECUTION_ENABLED = True

# Execution timeout (seconds) - prevents infinite loops
CODE_EXECUTION_TIMEOUT_SECONDS = 30

# Memory limit (bytes) - prevents memory exhaustion attacks
# Set to 6GB to allow large data analysis while leaving headroom in 10GB Lambda
# Note: Uses RLIMIT_AS (address space), applied during code execution only
CODE_EXECUTION_MEMORY_LIMIT_BYTES = 6 * 1024 * 1024 * 1024  # 6GB

# Output limits
CODE_EXECUTION_MAX_OUTPUT_CHARS = 50000  # Max stdout capture
CODE_EXECUTION_MAX_IMAGES = 2  # Max matplotlib figures to capture (reduced for simplicity)
CODE_EXECUTION_MAX_IMAGE_SIZE_MB = 5  # Max size per image
CODE_EXECUTION_MAX_WIDGET_SIZE_MB = 2  # Max size for Plotly HTML widgets

# Allowed imports (whitelist) - modules that can be used in generated code
# These are pre-imported and injected into the execution namespace
CODE_EXECUTION_ALLOWED_IMPORTS = {
    # Data manipulation
    "pandas", "numpy",
    # Scientific file formats
    "h5py",  # HDF5 file support (common in ML and scientific datasets)
    # Visualization
    "matplotlib", "seaborn", "plotly",
    # Statistical analysis
    "scipy", "sklearn", "statistics",
    # Utilities
    "json", "datetime", "time", "math", "io", "base64",
    "warnings", "collections", "itertools", "re",
    "random",  # For data sampling and shuffling
}

# Purpose-specific import restrictions
# Each purpose can only use a subset of allowed imports
CODE_EXECUTION_PURPOSE_IMPORTS = {
    "data_query": {"pandas", "numpy", "scipy", "statistics", "datetime", "time", "json", "math", "random", "itertools", "collections"},
    "visualization": {"pandas", "numpy", "matplotlib", "seaborn", "scipy", "io", "base64", "warnings", "datetime", "time", "itertools", "collections"},
    "widget": {"pandas", "numpy", "plotly", "json", "io", "datetime", "time", "itertools", "collections"},
    "simulation": {"pandas", "numpy", "matplotlib", "seaborn", "scipy", "io", "base64", "warnings", "random", "datetime", "time", "itertools", "collections"},
    "audit": {"pandas", "numpy", "scipy", "json", "statistics", "math", "random", "datetime", "time", "itertools", "collections"},
}

# Blocked patterns (deny list) - regex patterns that are always rejected
# These are checked BEFORE AST validation for fast rejection
CODE_EXECUTION_BLOCKED_PATTERNS = [
    # Block __import__ in any form (dict access, function call, etc.)
    r"__import__",
    # System command execution
    r"\bos\.system\b",
    r"\bos\.popen\b",
    r"\bsubprocess\b",
    r"\bos\.exec\w*\b",
    # Dynamic code execution
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    # File system manipulation (open() blocked via CODE_EXECUTION_BLOCKED_FUNCTIONS)
    r"\bos\.remove\b",
    r"\bos\.unlink\b",
    r"\bshutil\.rmtree\b",
    # Network access
    r"\bsocket\b",
    r"\burllib\b",
    r"\brequests\b",
    r"\bhttplib\b",
    # Credential/environment access
    r"\bos\.environ\b",
    r"\bos\.getenv\b",
    r"\bboto3\.client\b",
    r"\bboto3\.resource\b",
    r"\bboto3\b",                    # Block any boto3 usage
    # Pickle (arbitrary code execution risk)
    r"\bpickle\.loads?\b",
    r"\bpickle\b",                   # Block pickle module entirely
    # Image processing (not supported - use tabular data)
    r"\bPIL\b",
    r"\bImage\.open\b",
    r"\bfrom PIL\b",
    # Debugging modules (not needed - errors auto-captured)
    r"\btraceback\b",
]

# Blocked function names (checked during AST validation)
CODE_EXECUTION_BLOCKED_FUNCTIONS = {
    "eval", "exec", "compile", "__import__", "open",
    "input", "breakpoint", "help",
    # Note: exit/quit are NOT blocked - they're injected as no-ops in code_executor.py
    # This prevents unnecessary validation failures when LLM generates code with exit()
}

# Blocked attribute names (checked during AST validation)
CODE_EXECUTION_BLOCKED_ATTRIBUTES = {
    "system", "popen", "spawn", "fork", "exec",
    "environ", "getenv",
}

# Allowed builtins - minimal set for safe execution
# Everything else is blocked (no open, exec, eval, __import__, etc.)
CODE_EXECUTION_ALLOWED_BUILTINS = {
    # Core types
    "True", "False", "None",
    # Type constructors
    "bool", "int", "float", "str", "bytes", "list", "tuple", "dict", "set", "frozenset",
    # Iteration and sequences
    "len", "range", "enumerate", "zip", "map", "filter", "reversed", "sorted",
    # Aggregation
    "sum", "min", "max", "abs", "round", "pow", "divmod",
    # Type checking
    "isinstance", "issubclass", "type", "callable",
    # Attribute access
    "hasattr", "getattr", "setattr", "delattr",
    # Object utilities
    "id", "hash", "repr", "format", "print",
    # Iterators
    "iter", "next", "all", "any",
    # Exceptions (for error handling in generated code)
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "ZeroDivisionError",
}

# Plotly CDN allowlist for HTML sanitization (Widget Creator)
WIDGET_ALLOWED_SCRIPT_SOURCES = [
    r"^https://cdn\.plot\.ly/",
    r"^https://cdn\.plotly\.com/",
]

# HTML tags allowed in widget output (bleach configuration)
WIDGET_ALLOWED_TAGS = [
    "div", "script", "style",
    "svg", "g", "path", "rect", "circle", "text", "line", "polyline", "polygon",
    "defs", "clipPath", "use", "image", "tspan",
]

# HTML attributes allowed per tag (bleach configuration)
WIDGET_ALLOWED_ATTRIBUTES = {
    "div": ["id", "class", "style"],
    "script": ["src", "type"],
    "svg": ["width", "height", "viewBox", "xmlns", "class", "style"],
    "g": ["transform", "class", "style", "clip-path"],
    "path": ["d", "fill", "stroke", "stroke-width", "class", "style", "opacity"],
    "rect": ["x", "y", "width", "height", "fill", "stroke", "class", "style"],
    "circle": ["cx", "cy", "r", "fill", "stroke", "class", "style"],
    "text": ["x", "y", "text-anchor", "font-size", "fill", "class", "style", "transform"],
    "line": ["x1", "y1", "x2", "y2", "stroke", "stroke-width", "class", "style"],
    "*": ["id", "class"],  # Allow id/class on all elements
}

# Error message patterns to redact (security - don't leak paths/credentials)
ERROR_REDACTION_PATTERNS = [
    r"s3://[^\s]+",  # S3 paths
    r"arn:aws:[^\s]+",  # AWS ARNs
    r"\b\d{12}\b",  # AWS account IDs
    r"AKIA[A-Z0-9]{16}",  # AWS access keys
    r"(?i)password\s*[=:]\s*\S+",  # Passwords
    r"(?i)secret\s*[=:]\s*\S+",  # Secrets
    r"(?i)token\s*[=:]\s*\S+",  # Tokens
]


# =============================================================================
# Agentic Fix Loop Configuration (Layer 1)
# =============================================================================

# Time budget for the entire agentic tool-use loop (seconds).
# The analysis Lambda has 600s total; code generation uses ~60-90s.
AGENTIC_LOOP_TIME_BUDGET_SECONDS = 300

# Stop the loop if fewer than this many seconds remain before Lambda timeout.
AGENTIC_LOOP_MIN_REMAINING_SECONDS = 60

# Max tokens per LLM response within the agentic loop.
AGENTIC_LOOP_MAX_TOKENS = 16384

# Temperature for fix attempts — low for deterministic code fixes.
AGENTIC_LOOP_TEMPERATURE = 0.2

# Per-execution timeout within the loop (each execute_code call).
AGENTIC_LOOP_EXECUTION_TIMEOUT_SECONDS = 180

# Minimum figures from first-pass execution to skip the agentic loop.
# If code "succeeds" but produces fewer figures than this, or has too many
# internal errors, the agentic loop is triggered to improve the code.
AGENTIC_LOOP_MIN_FIGURES = 5

# Hard ceiling on loop iterations (secondary safety net alongside time budget).
AGENTIC_LOOP_MAX_ITERATIONS = 10

# Hint library configuration
HINT_LIBRARY_S3_KEY = "_system_/error_hints.json"
HINT_LIBRARY_MAX_ENTRIES = 200       # Trigger consolidation above this
HINT_LIBRARY_CONSOLIDATION_TARGET = 150  # Target after consolidation
HINT_LIBRARY_MAX_SOURCE_DATASETS = 20   # Cap on source_datasets per hint

# Telemetry: track agentic loop invocations to S3
AGENTIC_TELEMETRY_ENABLED = True
AGENTIC_TELEMETRY_S3_KEY = "_system_/agentic_loop_telemetry.json"
AGENTIC_TELEMETRY_MAX_RECORDS = 2000

# Telemetry: track metadata generation (Phase 1) outcomes to S3
METADATA_TELEMETRY_ENABLED = True
METADATA_TELEMETRY_S3_KEY = "_system_/metadata_generation_telemetry.json"
METADATA_TELEMETRY_MAX_RECORDS = 2000

# Redundancy consolidation — detect and restructure per-file visualization loops
CONSOLIDATION_LOOP_TIME_BUDGET_SECONDS = 120    # Max wall-clock time for consolidation pass
CONSOLIDATION_MIN_LOOP_SAVES = 2                # Minimum loop-based save calls to trigger
CONSOLIDATION_MAX_ITERATIONS = 6                 # Hard ceiling on consolidation loop iterations


# =============================================================================
# Development Configuration
# =============================================================================

# Development-specific settings
DEV_LOCAL_DOWNLOAD = True
DEV_VERBOSE_LOGGING = True
DEV_SKIP_LARGE_FILES = False

# Testing configuration
TEST_MODE = False
TEST_SAMPLE_SIZE = 5
TEST_MAX_FILES = 10
