from django.core.management.base import BaseCommand, CommandError
# from polls.models import Question as Poll # importing the required functions
import embeddingsearch

class Command(BaseCommand):
    help = "Read the uploaded document" # gives out a helpful description of the command functioning
    embeddingsearch()
    def add_arguments(self, parser):
        parser.add_argument("poll_ids", nargs="+", type=int)

    def handle(self, *args, **options):
        for poll_id in options["poll_ids"]:
            try:
                poll = Poll.objects.get(pk=poll_id)
            except Poll.DoesNotExist:
                raise CommandError('Poll "%s" does not exist' % poll_id)

            poll.opened = False
            poll.save()

            self.stdout.write(
                self.style.SUCCESS('Successfully closed poll "%s"' % poll_id)
            )