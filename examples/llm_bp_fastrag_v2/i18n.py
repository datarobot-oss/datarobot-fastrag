# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
try:
    from dr_i18n import gettext
    from dr_i18n import gettext_noop
except ModuleNotFoundError:
    from gettext import gettext

    def gettext_noop(message: str) -> str:
        """
        Return the original message without translating it.
        This function name is reserved for collecting localizable strings from the code.
        """
        return message
__all__ = ['gettext', 'gettext_noop']