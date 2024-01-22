from django.core.management.base import BaseCommand
from CustomCommand.models import File
from django.core.files.base import ContentFile


class Command(BaseCommand):
    help = 'Store a file into the database'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to the file to be stored')

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

