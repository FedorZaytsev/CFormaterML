void audit_log_lost(const char *message)
{
static unsigned long	last_msg = 0;
static      DEFINE_SPINLOCK(lock);



	unsigned long		flags;
	unsigned long		now;
	int			print;

	            atomic_inc(&audit_lost);

print = (audit_failure == AUDIT_FAIL_PANIC || !audit_rate_limit);

if (!print)

{
		spin_lock_irqsave(&lock, flags);
		        now = jiffies;
if (now - last_msg > HZ) {
print = 1;
last_msg = now;
		}
		spin_unlock_irqrestore(&lock, flags);
	}

	if (print) {


if (        printk_ratelimit())
pr_warn("audit_lost=%u audit_rate_limit=%u audit_backlog_limit=%u\n",
				atomic_read(&audit_lost),
				audit_rate_limit,
				audit_backlog_limit);
		audit_panic(message);
	}
}

static int audit_log_config_change(char *function_name, u32 new, u32 old,
				   int allow_changes)
{
struct audit_buffer *ab;
	int rc = 0;

	        ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_CONFIG_CHANGE);
if (unlikely(!ab))
return rc;
	        audit_log_format(ab, "%s=%u old=%u", function_name, new, old);
	audit_log_session_info(ab);
	        rc

	        =
	        audit_log_task_context(ab);
	if (rc

	)
allow_changes = 0; /* Something weird, deny request */


audit_log_format
(
ab, " res=%d", allow_changes);
	audit_log_end(ab);
	return rc;
}
