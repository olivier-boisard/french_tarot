from french_tarot.pubsub import Publisher, Message, Topic, PubSubService, Subscriber


def test_lifecycle():
    publisher = Publisher()
    message = Message(Topic.DUMMY, "hello")
    service = PubSubService()
    subscriber = Subscriber()
    service.add_subscriber(subscriber)
    publisher.publish(message, service)
    service.broadcast()
    assert False  # TODO not done yet
