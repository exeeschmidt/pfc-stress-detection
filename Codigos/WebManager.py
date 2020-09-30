import os
import webbrowser
import Datos


def buildHtml(video_path):
    filename_output = os.path.join(Datos.PATH_HTML, "index.html")
    f = open(filename_output, 'w')
    output = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stress Detection</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
    
            <!-- Bootstrap -->
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
    
            <!-- Video-JS -->
            <link rel="stylesheet" href="css/video-js.css">
            <script src="js/video.js"></script>
    
            <!-- Styles -->
            <link rel="stylesheet" href="css/styles.css">
        </head>
        <body>
            <header class="text-center">
              <h1>Detección de estrés</h1>
              <p>Resultados de la ejecución</p>
            </header>
            <main>
              <div class="container">
                <div class="row">
                  <div class="col-sm-6">
                    <h3>Video de entrada</h3>
                    <video class="fm-video video-js vjs-16-9 vjs-big-play-centered" data-setup="{}" controls id="fm-video">
                      <source src='""" + video_path + """'>
                    </video>
                  </div>
                  <div class="col-sm-6">
                    <iframe width="700" height="500" frameborder="0" seamless="seamless" scrolling="yes" src="resultado.html">
                    </iframe>
                  </div>
                </div>
              </div>
            </main>
            <script>
              var reproductor = videojs('fm-video', { fluid: true });
            </script>
        </body>
    </html>"""
    f.write(output)
    f.close()
    webbrowser.open_new_tab(filename_output)
    return
