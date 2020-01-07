from french_tarot.environment.core.announcements.poignee.poignee_announcement import PoigneeAnnouncement


class DoublePoigneeAnnouncement(PoigneeAnnouncement):
    @staticmethod
    def expected_length():
        return 13

    @staticmethod
    def bonus_points():
        return 30
