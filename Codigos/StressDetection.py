import click


@click.command()
@click.option('-v', '--video', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True))
def main(video):
    if str(video).find(".mp4") != -1:
        click.echo(click.format_filename(video))
    else:
        click.BadOptionUsage("video", "El archivo de video debe tener extensión .mp4", ctx=True)


if __name__ == '__main__':
    main()
