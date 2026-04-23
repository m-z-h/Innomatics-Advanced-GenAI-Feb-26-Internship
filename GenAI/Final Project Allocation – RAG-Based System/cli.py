"""
CLI Interface for RAG System
"""

import logging
import sys
from pathlib import Path
import click
from tabulate import tabulate

import config
from src.embeddings import create_embedding_provider
from src.vector_store import ChromaDBStore, InMemoryStore
from src.document_processor import DocumentProcessor
from src.chunking import create_chunker
from src.retrieval import Retriever
from src.query_processor import QueryProcessor
from src.llm_client import create_llm_client
from src.hitl import EscalationManager
from src.graph_engine import RAGWorkflowEngine

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


# Global system components
embedding_provider = None
vector_store = None
retriever = None
query_processor = None
llm_client = None
escalation_manager = None
workflow_engine = None


def init_system():
    """Initialize RAG system components"""
    global embedding_provider, vector_store, retriever, query_processor, llm_client, escalation_manager, workflow_engine
    
    logger.info("Initializing RAG system...")
    
    try:
        # Create embedding provider
        embedding_provider = create_embedding_provider(provider=config.EMBEDDING_PROVIDER)
        logger.info(f"✓ Embedding provider initialized ({config.EMBEDDING_PROVIDER})")
        
        # Create vector store
        vector_store = ChromaDBStore(
            embedding_provider=embedding_provider,
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION_NAME
        )
        logger.info(f"✓ Vector store initialized ({config.CHROMA_PERSIST_DIR})")
        
        # Create retriever
        retriever = Retriever(vector_store, embedding_provider)
        logger.info("✓ Retriever initialized")
        
        # Create query processor
        query_processor = QueryProcessor(retriever)
        logger.info("✓ Query processor initialized")
        
        # Create LLM client
        llm_client = create_llm_client(provider="openai")
        logger.info("✓ LLM client initialized")
        
        # Create escalation manager
        escalation_manager = EscalationManager()
        logger.info("✓ Escalation manager initialized")
        
        # Create workflow engine
        workflow_engine = RAGWorkflowEngine(
            query_processor=query_processor,
            retriever=retriever,
            llm_client=llm_client,
            escalation_manager=escalation_manager
        )
        logger.info("✓ Workflow engine initialized")
        
        logger.info("RAG system ready!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Initialization failed: {str(e)}")
        return False


@click.group()
def cli():
    """RAG-Based Customer Support Assistant CLI"""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
def upload(pdf_path):
    """Upload and index a PDF document"""
    
    if not init_system():
        sys.exit(1)
    
    try:
        click.echo(f"📄 Loading PDF: {pdf_path}")
        
        # Load PDF
        processor = DocumentProcessor()
        pdf_doc = processor.load_pdf(pdf_path)
        
        click.echo(f"✓ Loaded {pdf_doc.total_pages} pages")
        
        # Create chunks
        click.echo("🔄 Creating chunks...")
        chunker = create_chunker(
            strategy=config.CHUNKING_STRATEGY,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        
        chunks = []
        for text_block in pdf_doc.text_blocks:
            block_chunks = chunker.chunk(
                text=text_block.text,
                source_file=pdf_doc.filename,
                page_number=text_block.page_number
            )
            chunks.extend(block_chunks)
        
        click.echo(f"✓ Created {len(chunks)} chunks")
        
        # Add to vector store
        click.echo("📝 Indexing chunks...")
        vector_store.add_chunks(chunks)
        
        click.echo(f"✓ Successfully indexed {len(chunks)} chunks")
        
        # Show stats
        stats = vector_store.get_collection_stats()
        click.echo(f"\n📊 Collection Statistics:")
        for key, value in stats.items():
            click.echo(f"  {key}: {value}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--user-id", default="default", help="User ID")
def query(query, user_id):
    """Submit a query to the RAG system"""
    
    if not init_system():
        sys.exit(1)
    
    try:
        click.echo(f"🔍 Query: {query}")
        click.echo(f"👤 User: {user_id}")
        click.echo("-" * 80)
        
        # Execute workflow
        result = workflow_engine.execute(query=query, user_id=user_id)
        
        # Display results
        click.echo(f"\n📝 Response:")
        click.echo(result.response)
        
        click.echo(f"\n📊 Metadata:")
        click.echo(f"  Query ID: {result.query_id}")
        click.echo(f"  Confidence: {result.confidence:.1%}")
        click.echo(f"  Execution Time: {result.execution_time_ms:.0f}ms")
        click.echo(f"  Escalated: {'Yes' if result.is_escalated else 'No'}")
        
        if result.escalation_id:
            click.echo(f"  Escalation ID: {result.escalation_id}")
        
        if result.sources:
            click.echo(f"\n📚 Sources:")
            for source in result.sources:
                click.echo(f"  - {source['file']} (Page {source['page']}, Score: {source['score']:.2f})")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--pending", is_flag=True, default=True, help="Show pending escalations")
@click.option("--limit", default=10, help="Max number to show")
def escalations(pending, limit):
    """View escalations queue"""
    
    if not init_system():
        sys.exit(1)
    
    try:
        status = "pending" if pending else "resolved"
        escalations_list = escalation_manager.get_escalation_queue(status=status, limit=limit)
        
        if not escalations_list:
            click.echo(f"No {status} escalations found.")
            return
        
        # Prepare table data
        table_data = []
        for esc in escalations_list:
            table_data.append([
                esc.escalation_id[:8],
                esc.original_query[:50],
                esc.escalation_reason,
                f"{esc.confidence_score:.2f}",
                esc.created_at.strftime("%H:%M")
            ])
        
        headers = ["ID", "Query", "Reason", "Confidence", "Created"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Show stats
        click.echo(f"\nTotal: {len(escalations_list)}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def stats():
    """Show system statistics"""
    
    if not init_system():
        sys.exit(1)
    
    try:
        stats_data = escalation_manager.get_feedback_stats()
        
        click.echo("\n📊 System Statistics:")
        click.echo("=" * 50)
        
        for key, value in stats_data.items():
            if isinstance(value, float):
                click.echo(f"{key}: {value:.2%}" if key.endswith("rate") else f"{key}: {value:.2f}")
            elif isinstance(value, dict):
                click.echo(f"\n{key}:")
                for k, v in value.items():
                    click.echo(f"  {k}: {v}")
            else:
                click.echo(f"{key}: {value}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def health():
    """Check system health"""
    
    try:
        click.echo("🏥 Checking system health...")
        
        if init_system():
            click.echo("✓ All components initialized successfully")
            
            # Check vector store
            try:
                stats = vector_store.get_collection_stats()
                click.echo(f"✓ Vector store: {stats['total_chunks']} chunks")
            except Exception as e:
                click.echo(f"⚠ Vector store: {str(e)}")
            
            click.echo("\n✓ System is healthy!")
        else:
            click.echo("✗ System initialization failed")
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
