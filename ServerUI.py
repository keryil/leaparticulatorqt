import Constants, sys
from LeapServer import LeapServerFactory
if "twisted.internet.reactor" not in sys.modules:
    from twisted.internet import gtk2reactor
    gtk2reactor.install()
    # from twisted.internet import reactor
from twisted.internet import protocol, reactor

from twisted.internet.endpoints import TCP4ServerEndpoint
# from twisted.internet.protocol import ServerFactory
import pygtk
pygtk.require("2.0")
import gtk
from twisted.python import log

class ServerUI(object):
    def __init__(self):
        self.glade_file = "ServerWindow.glade"
        self.builder = builder = gtk.Builder()
        builder.add_from_file(self.glade_file)

        self.mainWindow = builder.get_object("serverWindow")
        self.listStore = builder.get_object("clientsListStore")

        renderer = gtk.CellRendererText()
        column = gtk.TreeViewColumn("Title", renderer, text=0)
        tree = builder.get_object("clientsTreeView")
        tree.append_column(column)
        self.mainWindow.show_all()

        self.endpoint = endpoint = TCP4ServerEndpoint(reactor, Constants.leap_port)
        endpoint.listen(LeapServerFactory(ui=self))

    def connectionMade(self, server):
        self.listStore.append([server.other_end])#, server.factory.mode])
        self.mainWindow.queue_draw()
        log.msg("Connection with %s made!" % server.other_end)

    def connectionLost(self, server, reason):
        log.msg("Connection with %s lost because %s!" % (server.other_end,reason))


if __name__ == '__main__':
    ui = ServerUI()
    gtk.main()