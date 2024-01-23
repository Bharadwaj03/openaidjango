# myapp/management/commands/run_python_file.py
from django.core.management.base import BaseCommand
from CustomCommand.embeddingsearch import search_with_embeddings
from django.core.files.base import ContentFile
from CustomCommand.models import File

class Command(BaseCommand):
    help = 'Run an embedding-based search using ChromaDB and OpenAI'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str,  help='Path to the file to be stored')
        parser.add_argument('search_string', type=str, help='String to search with embeddings')
        

    def handle(self, *args, **options):
        file_path = options['file_path']
        try:
            with open(file_path, 'rb') as file:
                file_content = file.read()
                file_name = file_path.split('/')[-1]
                content_file = ContentFile(file_content)
                new_file = File(name=file_name, content=content_file)
                new_file.save()
                self.stdout.write(self.style.SUCCESS(f'Successfully stored file "{file_name}" in the database'))
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f'File not found at path: {file_path}'))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'Error storing file: {str(e)}'))
        
        search_string = options['search_string']

        # Call the function with embedding-based search
        result = search_with_embeddings(search_string)

        if result:
            self.stdout.write(self.style.SUCCESS(f'Result "{search_string}" found using embeddings'))
        else:
            self.stdout.write(self.style.ERROR(f'Result "{search_string}" not found using embeddings'))
