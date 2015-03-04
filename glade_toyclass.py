

#!/usr/bin/env python

import sys, pygtk, gtk
# try:
#     import pygtk
#     pygtk.require("2.0")
# except:
#     pass
# try:
#     import gtk
#     import gtk.glade
# except:
#     sys.exit(333)

class HellowWorldGTK:
    """This is an Hello World GTK application"""

    def __init__(self):
        
        #Set the Glade file
        self.gladefile = "ClientWindow.glade"  
        self.wTree = gtk.glade.XML(self.gladefile) 
        print "Oh be"
        
        #Get the Main Window, and connect the "destroy" event
        self.window = self.wTree.get_widget("MainWindow")
        if (self.window):
            self.window.connect("destroy", gtk.main_quit)



if __name__ == "__main__":
    print "Start"
    hwg = HellowWorldGTK()
    print "Go on"
    gtk.main()
    print "Die"

