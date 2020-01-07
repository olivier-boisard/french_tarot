from french_tarot.environment.core.announcements.poignee.poignee_announcement import PoigneeAnnouncement


class SimplePoigneeAnnouncement(PoigneeAnnouncement):

    @staticmethod
    def expected_length():
        return 10

    @staticmethod
    def bonus_points():
        return 20
